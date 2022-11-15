import os

import tensorflow as tf
import numpy as np
from numpy.random import choice
import pandas as pd

from proteolizardalgo.utility import preprocess_max_quant_sequence
from proteolizardalgo.chemistry import get_mono_isotopic_weight, ccs_to_one_over_reduced_mobility, MASS_PROTON

from enum import Enum
from abc import ABC, abstractmethod


class ENZYME(Enum):
    TRYPSIN = 1


class ORGANISM(Enum):
    HOMO_SAPIENS = 1


class Enzyme(ABC):
    def __init__(self, name: ENZYME):
        self.name = name

    @abstractmethod
    def calculate_cleavages(self, sequence: str) -> np.array:
        pass

    def digest(self, sequence: str, missed_cleavages: int, min_length) -> list[str]:
        pass


class Trypsin(Enzyme):
    def __init__(self, name: ENZYME = ENZYME.TRYPSIN):
        super().__init__(name)
        self.name = name

    def calculate_cleavages(self, sequence):

        cut_sites = [0]

        for i in range(len(sequence) - 1):
            # cut after every 'K' or 'R' that is not followed by 'P'
            if ((sequence[i] == 'K') or (sequence[i] == 'R')) and sequence[i + 1] != 'P':
                cut_sites += [i + 1]

        cut_sites += [len(sequence)]

        return np.array(cut_sites)

    def __repr__(self):
        return f'Enzyme(name: {self.name.name})'

    def digest(self, sequence, missed_cleavages=0, min_length=7):
        assert 0 <= missed_cleavages <= 2, f'Number of missed cleavages might be between 0 and 2, was: {missed_cleavages}'

        cut_sites = self.calculate_cleavages(sequence)
        pairs = np.c_[cut_sites[:-1], cut_sites[1:]]

        # TODO: implement
        if missed_cleavages == 1:
            pass
        if missed_cleavages == 2:
            pass

        dig_list = []

        for s, e in pairs:
            dig_list += [sequence[s: e]]

        # pair sequence digests with indices
        wi = zip(dig_list, pairs)
        # filter out short sequences and clean index display
        wi = map(lambda p: (p[0], p[1][0], p[1][1]), filter(lambda s: len(s[0]) >= min_length, wi))

        return list(map(lambda e: {'sequence': e[0], 'start': e[1], 'end': e[2]}, wi))


class PeptideDigest:
    def __init__(self, data: pd.DataFrame, organism: ORGANISM, enzyme: ENZYME):
        self.data = data
        self.organism = organism
        self.enzyme = enzyme

    def __repr__(self):
        return f'PeptideMix(Sample: {self.organism.name}, Enzyme: {self.enzyme.name}, number peptides: {self.data.shape[0]})'


class ProteinSample:

    def __init__(self, data: pd.DataFrame, name: ORGANISM):
        self.data = data
        self.name = name

    def digest(self, enzyme: Enzyme, missed_cleavages: int = 0, min_length: int = 7) -> PeptideDigest:

        digest = self.data.apply(lambda r: enzyme.digest(r['sequence'], missed_cleavages, min_length), axis=1)

        V = zip(self.data['id'].values, digest.values)

        r_list = []

        for (gene, peptides) in V:
            for pep in peptides:
                if pep['sequence'].find('X') == -1:
                    pep['id'] = gene
                    pep['sequence'] = '_' + pep['sequence'] + '_'
                    pep['sequence-tokenized'] = preprocess_max_quant_sequence(pep['sequence'])
                    pep['mass-theoretical'] = get_mono_isotopic_weight(pep['sequence-tokenized'])
                    r_list.append(pep)

        return PeptideDigest(pd.DataFrame(r_list), self.name, enzyme.name)

    def __repr__(self):
        return f'ProteinSample(Organism: {self.name.name})'


class LiquidChromatography(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_retention_times(self, sequences: list[str]) -> np.array:
        pass


class NeuralChromatography(LiquidChromatography):

    def __init__(self, model_path: str, tokenizer: tf.keras.preprocessing.text.Tokenizer):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = tokenizer

    def sequences_to_tokens(self, sequences: np.array) -> np.array:
        print('tokenizing sequences...')
        seq_lists = [list(s) for s in sequences]
        tokens = self.tokenizer.texts_to_sequences(seq_lists)
        tokens_padded = tf.keras.preprocessing.sequence.pad_sequences(tokens, 50, padding='post')
        return tokens_padded

    def sequences_tf_dataset(self, sequences: np.array, batched: bool = True, bs: int = 2048) -> tf.data.Dataset:
        tokens = self.sequences_to_tokens(sequences)
        print('generating tf dataset...')
        pseudo_target = np.expand_dims(np.zeros_like(tokens[:, 0]), axis=1)

        if batched:
            return tf.data.Dataset.from_tensor_slices((tokens, pseudo_target)).batch(bs)
        return tf.data.Dataset.from_tensor_slices((tokens, pseudo_target))

    def get_retention_times(self, data: pd.DataFrame) -> np.array:
        ds = self.sequences_tf_dataset(data['sequence-tokenized'])
        print('predicting irts...')
        return self.model.predict(ds)


class IonSource(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def ionize(self, data: pd.DataFrame, allowed_charges: list = [1, 2, 3, 4, 5]) -> np.array:
        pass


class RandomIonSource(IonSource):
    def __init__(self):
        super().__init__()

    def ionize(self, data, allowed_charges: list = [1, 2, 3, 4], p: list = [0.1, 0.5, 0.3, 0.1]):
        return choice(allowed_charges, data.shape[0], p=p)


class IonMobilitySeparation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_mobilities_and_ccs(self, data: pd.DataFrame) -> np.array:
        pass


class NeuralMobilitySeparation(IonMobilitySeparation):

    def __init__(self, model_path: str, tokenizer: tf.keras.preprocessing.text.Tokenizer):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = tokenizer

    def sequences_to_tokens(self, sequences: np.array) -> np.array:
        print('tokenizing sequences...')
        seq_lists = [list(s) for s in sequences]
        tokens = self.tokenizer.texts_to_sequences(seq_lists)
        tokens_padded = tf.keras.preprocessing.sequence.pad_sequences(tokens, 50, padding='post')
        return tokens_padded

    def sequences_tf_dataset(self, mz: np.array, charges: np.array, sequences: np.array,
                             batched: bool = True, bs: int = 2048) -> tf.data.Dataset:
        tokens = self.sequences_to_tokens(sequences)
        mz = np.expand_dims(mz, 1)
        c = tf.one_hot(charges - 1, depth=4)
        print('generating tf dataset...')
        pseudo_target = np.expand_dims(np.zeros_like(tokens[:, 0]), axis=1)

        if batched:
            return tf.data.Dataset.from_tensor_slices(((mz, c, tokens), pseudo_target)).batch(bs)
        return tf.data.Dataset.from_tensor_slices(((mz, c, tokens), pseudo_target))

    def get_mobilities_and_ccs(self, data: pd.DataFrame) -> np.array:
        ds = self.sequences_tf_dataset(data['mz'], data['charge'], data['sequence-tokenized'])

        mz = data['mz'].values

        print('predicting mobilities...')
        ccs, _ = self.model.predict(ds)
        one_over_k0 = ccs_to_one_over_reduced_mobility(np.squeeze(ccs), mz, data['charge'].values)

        return np.c_[ccs, one_over_k0]