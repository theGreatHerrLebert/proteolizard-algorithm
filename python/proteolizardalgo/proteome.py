
import numpy as np
import pandas as pd

from proteolizardalgo.utility import preprocess_max_quant_sequence
from proteolizardalgo.chemistry import get_mono_isotopic_weight

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


class ProteomicsExperimentSample:
    """
    A proteomics experiment could analyze e.g. whole proteins or
    peptide digestions.
    """
    def __init__(self, input:PeptideDigest):
        self.data: pd.DataFrame = input.data
        self._input = input
