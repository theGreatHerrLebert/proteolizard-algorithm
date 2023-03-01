from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
import pandas as pd

from proteolizardalgo.chemistry import  ccs_to_one_over_reduced_mobility
from proteolizardalgo.proteome import ProteomicsExperimentSampleSlice

class Chromatography(ABC):
    def __init__(self):
        self._apex_model = None
        self._profile_model = None
        self._frame_length = None
        self._gradient_length = None

    @property
    def frame_length(self):
        return self._frame_length

    @frame_length.setter
    def frame_length(self, milliseconds: int):
        self._frame_length = milliseconds

    @property
    def gradient_length(self):
        return self._gradient_length/(60*1000)

    @gradient_length.setter
    def gradient_length(self, minutes: int):
        self._gradient_length = minutes*60*1000

    @property
    def num_frames(self):
        return self._gradient_length//self._frame_length

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
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

    @abstractmethod
    def irt_to_frame_id(self):
        pass

class LiquidChromatography(Chromatography):
    def __init__(self):
        super().__init__()

    def run(self, sample: ProteomicsExperimentSampleSlice):
        # retention time apex simulation
        retention_time_apex = self._apex_model.get_retention_times(sample)
        # in irt
        sample.add_prediction("retention_time_apex", retention_time_apex)
        # in frame id
        sample.add_prediction("frame_apex", self.irt_to_frame_id(retention_time_apex))

        # profile simulation
        retention_profile = self._profile_model.get_retention_profile(sample)
        sample.add_prediction("retention_profile", retention_profile)

    def irt_to_frame_id_vector( irt, max_frame=66000, irt_min=-30, irt_max=170):
        spacing = np.linspace(irt_min, irt_max, max_frame).reshape((-1,1)) + 1
        irt = irt.reshape((1,-1))
        return np.argmin(np.abs(spacing - irt), axis=0)



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
    def ionize(self, sample: ProteomicsExperimentSampleSlice):
        pass

class ElectroSpray(IonSource):
    def __init__(self):
        super().__init__()

    def ionize(self, sample: ProteomicsExperimentSampleSlice):
        pass


class IonMobilitySeparation(ABC):
    def __init__(self):
        self._apex_model = None
        self._profile_model = None
        self._num_scans = None
        self._scan_time = None

    @property
    def num_scans(self):
        return self._num_scans

    @num_scans.setter
    def num_scans(self, number:int):
        self._num_scans = number

    @property
    def scan_time(self):
        return self._scan_time

    @scan_time.setter
    def scan_time(self, microseconds:int):
        self._scan_time = microseconds

    @property
    def apex_model(self):
        return self._apex_model

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
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

    @abstractmethod
    def im_to_scan(self):
        pass

class TrappedIon(IonMobilitySeparation):

    def __init__(self):
        super().__init__()

    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

    def im_to_scan(self, inv_mob, slope=-880.57513791, intercept=1454.29035506):
        return int(np.round(inv_mob * slope + intercept))


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
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

class TOF(MzSeparation):
    def __init__(self):
        super().__init__()

    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass
class ChromatographyApexModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_retention_times(self, input: ProteomicsExperimentSampleSlice):
        pass

class ChromatographyProfileModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_retention_profile(self, input: ProteomicsExperimentSampleSlice):
        pass

class DummyChromatographyProfileModel(ChromatographyProfileModel):

    def __init__(self):
        super().__init__()

    def get_retention_profile(self, input: ProteomicsExperimentSampleSlice):
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

    def get_retention_times(self, input: ProteomicsExperimentSampleSlice) -> np.array:
        data = input.data
        ds = self.sequences_tf_dataset(data['sequence-tokenized'])
        print('predicting irts...')
        return self.model.predict(ds)


class IonMobilityApexModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_mobilities_and_ccs(self, input: ProteomicsExperimentSampleSlice):
        pass

class IonMobilityProfileModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_mobility_profile(self, input: ProteomicsExperimentSampleSlice):
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

    def get_mobilities_and_ccs(self, input: ProteomicsExperimentSampleSlice) -> np.array:
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

    def get_mobility_profile(self, input: ProteomicsExperimentSampleSlice):
        return None

class IonizationModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def ionize(self, input:ProteomicsExperimentSampleSlice):
        pass

class RandomIonSource(IonizationModel):
    def __init__(self):
        super().__init__()

    def ionize(self, ProteomicsExperimentSampleSlice, allowed_charges: list = [1, 2, 3, 4], p: list = [0.1, 0.5, 0.3, 0.1]):
        data = ProteomicsExperimentSampleSlice.data
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