from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
import pandas as pd

from proteolizardalgo.chemistry import  ccs_to_one_over_reduced_mobility
from proteolizardalgo.proteome import ProteomicsExperimentSample

class ChromatographyApexModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_retention_times(self, input: ProteomicsExperimentSample):
        pass

class ChromatographyProfileModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_retention_profile(self, input: ProteomicsExperimentSample):
        pass

class DummyChromatographyProfileModel(ChromatographyProfileModel):

    def __init__(self):
        super().__init__()

    def get_retention_profile(self, input: ProteomicsExperimentSample):
        return None


class NeuralChromatographyApex(ChromatographyApexModel):

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

    def get_retention_times(self, input: ProteomicsExperimentSample) -> np.array:
        data = input.data
        ds = self.sequences_tf_dataset(data['sequence-tokenized'])
        print('predicting irts...')
        return self.model.predict(ds)


class IonMobilityApexModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_mobilities_and_ccs(self, input: ProteomicsExperimentSample):
        pass

class IonMobilityProfileModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_mobility_profile(self, input: ProteomicsExperimentSample):
        pass

class NeuralMobilityApex(IonMobilityApexModel):

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

    def get_mobilities_and_ccs(self, input: ProteomicsExperimentSample) -> np.array:
        data = input.data
        ds = self.sequences_tf_dataset(data['mz'], data['charge'], data['sequence-tokenized'])

        mz = data['mz'].values

        print('predicting mobilities...')
        ccs, _ = self.model.predict(ds)
        one_over_k0 = ccs_to_one_over_reduced_mobility(np.squeeze(ccs), mz, data['charge'].values)

        return np.c_[ccs, one_over_k0]

class DummyIonMobilityProfileModel(IonMobilityProfileModel):
    def __init__(self):
        super().__init__()

    def get_mobility_profile(self, input: ProteomicsExperimentSample):
        return None

class IonizationModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def ionize(self, input:ProteomicsExperimentSample):
        pass

class RandomIonSource(IonizationModel):
    def __init__(self):
        super().__init__()

    def ionize(self, ProteomicsExperimentSample, allowed_charges: list = [1, 2, 3, 4], p: list = [0.1, 0.5, 0.3, 0.1]):
        data = ProteomicsExperimentSample.data
        return np.random.choice(allowed_charges, data.shape[0], p=p)

class MzSeparationModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_spectrum(self):
        pass

class TOFModel(MzSeparationModel):
    def __init__(self):
        super().__init__()

    def get_spectrum(self):
        pass