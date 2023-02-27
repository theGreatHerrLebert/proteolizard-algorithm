from abc import ABC, abstractmethod
import pandas as pd

from proteolizardalgo.proteome import PeptideDigest
import proteolizardalgo.hardware as hardware

class ProteomicsExperimentSample:
    def __init__(self):
        self._data = None
        self._input = None

    def load(self, input:PeptideDigest):
        self._input = input
        self._data = input.data
class ProteomicsExperiment(ABC):
    def __init__(self):
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
    def add_sample(self, sample_data: PeptideDigest):
        pass

    @abstractmethod
    def run(self):
        pass


class TimsTOFExperiment(ProteomicsExperiment):
    def __init__(self):
        super().__init__()

    def add_sample(self, sample_data:PeptideDigest, reduce:bool = False, sample_size: int = 1000):
        if reduce:
            self.sample_signal = sample_data.sample(sample_size)
        else:
            self.sample_signal = sample_data

    def run(self):
        pass