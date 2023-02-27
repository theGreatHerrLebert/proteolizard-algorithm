from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
import pandas as pd

from proteolizardalgo.chemistry import  ccs_to_one_over_reduced_mobility



class LiquidChromatography(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_retention_times(self, sequences: list[str]) -> np.array:
        pass


class IonSource(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_ionization_model(self):
        pass

    @abstractmethod
    def ionize(self, data: pd.DataFrame, allowed_charges: list = [1, 2, 3, 4, 5]) -> np.array:
        pass

class ElectroSpray(IonSource):
    def __init__(self):
        pass

    def set_ionization_model(self):
        pass

    def ionize(self, data: pd.DataFrame, allowed_charges: list = [1, 2, 3, 4, 5]) -> np.array:
        pass


class IonMobilitySeparation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_apex_model(self):
        pass

    @abstractmethod
    def set_profile_model(self):
        pass

    @abstractmethod
    def get_mobilities_and_ccs(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def get_mobility_profile(self, data:pd.DataFrame):
        pass

class TrappedIon(IonMobilitySeparation):

    def __init__(self):
        super().__init__()


class MzSeparation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_spectrum(self):
        pass

class TOF(MzSeparation):
    def __init__(self):
        super().__init__()

    def get_spectrum(self):
        pass