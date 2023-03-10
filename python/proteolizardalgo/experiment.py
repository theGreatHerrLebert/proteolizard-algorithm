import os
from abc import ABC, abstractmethod
import pandas as pd

from proteolizardalgo.proteome import PeptideDigest, ProteomicsExperimentSampleSlice, ProteomicsExperimentDatabaseHandle
import proteolizardalgo.hardware_models as hardware

class ProteomicsExperiment(ABC):
    def __init__(self, path: str):
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.database = ProteomicsExperimentDatabaseHandle(path)
        self.loaded_sample = None

        # signal noise discrimination
        self.sample_signal = None
        self.noise_signal = None

        # hardware methods
        self._lc_method = None
        self._ionization_method = None
        self._ion_mobility_separation_method = None
        self._mz_separation_method = None

    @property
    def lc_method(self):
        return self._lc_method

    @lc_method.setter
    def lc_method(self, method: hardware.LiquidChromatography):
        self._lc_method = method

    @property
    def ionization_method(self):
        return self._ionization_method

    @ionization_method.setter
    def ionization_method(self, method: hardware.IonSource):
        self._ionization_method = method

    @property
    def ion_mobility_separation_method(self):
        return self._ion_mobility_separation_method

    @ion_mobility_separation_method.setter
    def ion_mobility_separation_method(self, method: hardware.IonMobilitySeparation):
        self._ion_mobility_separation_method = method

    @property
    def mz_separation_method(self):
        return self._mz_separation_method

    @mz_separation_method.setter
    def mz_separation_method(self, method: hardware.MzSeparation):
        self._mz_separation_method = method

    @abstractmethod
    def load_sample(self, sample: PeptideDigest):
        self.database.push("PeptideDigest",sample)

    @abstractmethod
    def run(self):
        pass


class TimsTOFExperiment(ProteomicsExperiment):
    def __init__(self, path:str):
        super().__init__(path)

    def load_sample(self, sample: PeptideDigest):
        return super().load_sample(sample)

    def run(self, chunk_size: int = 1000):
        # load bulks of data here as dataframe if necessary
        for data_chunk in self.database.load_chunks(chunk_size):
            self.lc_method.run(data_chunk)
            self.ionization_method.run(data_chunk)
            self.ion_mobility_separation_method.run(data_chunk)
            self.database.update(data_chunk)
