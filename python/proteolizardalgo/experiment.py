from abc import ABC, abstractmethod
import pandas as pd

from proteolizardalgo.proteome import PeptideDigest
import proteolizardalgo.hardware as hardware

class ProteomicsExperiment(ABC):
    def __init__(self):
        self.sample_signal = None
        self.noise_signal = None

    @abstractmethod
    def add_sample(self, sample_data: PeptideDigest):
        pass

    @abstractmethod
    def set_lc_method(self, Liquid):
        pass

    @abstractmethod
    def set_ion_source_method(self, ion_source: hardware.IonSource):
        pass

    @abstractmethod
    def set_ion_mobility_separation_method(self, ion_mobility_separation: hardware.IonMobilitySeparation):
        pass

    @abstractmethod
    def set_mz_separation_method(self, mz_separation: hardware.MzSeparation):
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

