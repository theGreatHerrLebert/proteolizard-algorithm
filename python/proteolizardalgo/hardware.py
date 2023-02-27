from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
import pandas as pd

from proteolizardalgo.chemistry import  ccs_to_one_over_reduced_mobility
import proteolizardalgo.hardware_models as models
from proteolizardalgo.experiment import ProteomicsExperimentSample

class Chromatography(ABC):
    def __init__(self):
        self._apex_model = None
        self._profile_model = None

    @property
    def apex_model(self):
        return self._apex_model

    @apex_model.setter
    def apex_model(self, model:models.ChromatographyApexModel):
        self._apex_model = model

    @property
    def profile_model(self):
        return self._profile_model

    @profile_model.setter
    def profile_model(self, model:models.ChromatographyProfileModel):
        self._profile_model = model

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSample):
        pass

class LiquidChromatography(Chromatography):
    def __init__(self):
        super().__init__()

    def run(self, sample: ProteomicsExperimentSample):
        pass

class IonSource(ABC):
    def __init__(self):
        self._ionization_model = None

    @property
    def ionization_model(self):
        return self._ionization_model

    @ionization_model.setter
    def ionization_model(self, model: models.IonizationModel):
        self._ionization_model = model

    @abstractmethod
    def ionize(self, sample: ProteomicsExperimentSample):
        pass

class ElectroSpray(IonSource):
    def __init__(self):
        super().__init__()

    def ionize(self, sample: ProteomicsExperimentSample):
        pass


class IonMobilitySeparation(ABC):
    def __init__(self):
        self._apex_model = None
        self._profile_model = None

    @property
    def apex_model(self):
        return self.apex_model

    @apex_model.setter
    def apex_model(self, model: models.IonMobilityApexModel):
        self._apex_model = model

    @property
    def profile_model(self):
        return self._profile_model

    @profile_model.setter
    def profile_model(self, model: models.IonMobilityProfileModel):
        self._profile_model = model

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSample):
        pass

class TrappedIon(IonMobilitySeparation):

    def __init__(self):
        super().__init__()

    def run(self, sample: ProteomicsExperimentSample):
        pass


class MzSeparation(ABC):
    def __init__(self):
        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: models.MzSeparationModel):
        self._model = model

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSample):
        pass

class TOF(MzSeparation):
    def __init__(self):
        super().__init__()

    def run(self, sample: ProteomicsExperimentSample):
        pass