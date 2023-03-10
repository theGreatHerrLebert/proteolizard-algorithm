from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import tensorflow as tf
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.stats import exponnorm, norm

from proteolizardalgo.chemistry import  ccs_to_one_over_reduced_mobility
from proteolizardalgo.proteome import ProteomicsExperimentSampleSlice
from proteolizardalgo.feature import RTProfile, ScanProfile, ChargeProfile
from proteolizardalgo.utility import ExponentialGaussianDistribution as emg


class Device(ABC):
    def __init__(self, name:str):
        self.name = name

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Device):
        pass

class Chromatography(Device):
    def __init__(self, name:str="ChromatographyDevice"):
        super().__init__(name)
        self._apex_model = None
        self._profile_model = None
        self._irt_to_rt_converter = None
        self._frame_length = 1200
        self._gradient_length = 120*60*1000 # 120 minutes in miliseconds

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
    def apex_model(self, model: ChromatographyApexModel):
        self._apex_model = model

    @property
    def profile_model(self):
        return self._profile_model

    @property
    def irt_to_rt_converter(self):
        return self._irt_to_rt_converter

    @irt_to_rt_converter.setter
    def irt_to_rt_converter(self, converter:callable):
        self._irt_to_rt_converter = converter

    @profile_model.setter
    def profile_model(self, model:ChromatographyProfileModel):
        self._profile_model = model

    def irt_to_frame_id(self, irt: float):
        return self.rt_to_frame_id(self.irt_to_rt(irt))

    @abstractmethod
    def rt_to_frame_id(self, rt: float):
        pass

    def irt_to_rt(self, irt):
        return self._irt_to_rt_converter(irt)

    def frame_time_interval(self, frame_id:ArrayLike):
        s = (frame_id-1)*self.frame_length
        e = frame_id*self.frame_length
        return np.stack([s, e], axis = 1)

    def frame_time_middle(self, frame_id: ArrayLike):
        return np.mean(self.frame_time_interval(frame_id),axis=1)

class LiquidChromatography(Chromatography):
    def __init__(self, name: str = "LiquidChromatographyDevice"):
        super().__init__(name)

    def run(self, sample: ProteomicsExperimentSampleSlice):
        # retention time apex simulation
        retention_time_apex = self._apex_model.simulate(sample, self)
        # in irt and rt
        sample.add_simulation("simulated_irt_apex", retention_time_apex)
        # in frame id
        sample.add_simulation("simulated_frame_apex", self.irt_to_frame_id(retention_time_apex))

        # profile simulation

        retention_profile = self._profile_model.simulate(sample, self)
        sample.add_simulation("simulated_frame_profile", retention_profile)

    def rt_to_frame_id(self,  rt_seconds: ArrayLike):
        rt_seconds = np.asarray(rt_seconds)
        # first frame is completed not at 0 but at frame_length
        frame_id = (rt_seconds/self.frame_length*1000).astype(np.int64)+1
        return frame_id

class ChromatographyApexModel(Model):
    def __init__(self):
        self._device = None

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Chromatography) -> NDArray[np.float64]:
        pass

class ChromatographyProfileModel(Model):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Chromatography) -> List[RTProfile]:
        pass

class EMGChromatographyProfileModel(ChromatographyProfileModel):

    def __init__(self):
        super().__init__()

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Chromatography) -> List[RTProfile]:
        mus = sample.peptides["simulated_irt_apex"].values
        frames = sample.peptides["simulated_frame_apex"].values
        ϵ = device.frame_length
        profile_list = []
        for mu, frame in zip(mus,frames):
            σ = 1 # must be sampled
            λ = 1 # must be sampled
            K = 1/(σ*λ)
            μ = device.irt_to_rt(mu)
            model_params = { "sigma":σ,
                             "lambda":λ,
                             "mu":μ,
                             "name":"EMG"
                            }

            emg = exponnorm(loc=μ, scale=σ, K = K)
            # start and end value (in retention times)
            s_rt, e_rt = emg.ppf([0.05,0.95])
            # as frames
            s_frame, e_frame = device.rt_to_frame_id(s_rt), device.rt_to_frame_id(e_rt)

            profile_frames = np.arange(s_frame,e_frame+1)
            profile_middle_times = device.frame_time_middle(profile_frames)
            profile_rel_intensities = emg.pdf(profile_middle_times)* ϵ

            profile_list.append(RTProfile(profile_frames,profile_rel_intensities,model_params))
        return profile_list




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

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Chromatography) ->  NDArray[np.float64]:
        data = sample.peptides
        ds = self.sequences_tf_dataset(data['sequence-tokenized'])
        print('predicting irts...')
        return self.model.predict(ds)

class IonSource(Device):
    def __init__(self, name:str ="IonizationDevice"):
        super().__init__(name)
        self._ionization_model = None

    @property
    def ionization_model(self):
        return self._ionization_model

    @ionization_model.setter
    def ionization_model(self, model: IonizationModel):
        self._ionization_model = model

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

class ElectroSpray(IonSource):
    def __init__(self, name:str ="ElectrosprayDevice"):
        super().__init__(name)

    def run(self, sample: ProteomicsExperimentSampleSlice):
        charge_profiles = self.ionization_model.simulate(sample, self)
        sample.add_simulation("simulated_charge_profile", charge_profiles)

class IonizationModel(Model):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample:ProteomicsExperimentSampleSlice, device: IonSource) -> NDArray:
        pass

class RandomIonSource(IonizationModel):
    def __init__(self):
        super().__init__()
        self._charge_probabilities =  np.array([0.1, 0.5, 0.3, 0.1])
        self._allowed_charges =  np.array([1, 2, 3, 4], dtype=np.int8)

    @property
    def allowed_charges(self):
        return self._allowed_charges

    @allowed_charges.setter
    def allowed_charge(self, charges: ArrayLike):
        self._allowed_charges = np.asarray(charges, dtype=np.int8)

    @property
    def charge_probabilities(self):
        return self._charge_probabilities

    @charge_probabilities.setter
    def charge_probabilities(self, probabilities: ArrayLike):
        self._charge_probabilities = np.asarray(probabilities)

    def simulate(self, ProteomicsExperimentSampleSlice, device: IonSource) -> List[ChargeProfile]:
        if self.charge_probabilities.shape != self.allowed_charges.shape:
            raise ValueError("Number of allowed charges must fit to number of probabilities")

        data = ProteomicsExperimentSampleSlice.peptides
        charge = np.random.choice(self.allowed_charges, data.shape[0], p=self.charge_probabilities)
        rel_intensity = np.ones_like(charge)
        charge_profiles = []
        for c,i in zip(charge, rel_intensity):
            charge_profiles.append(ChargeProfile([c],[i],model_params={"name":"RandomIonSource"}))
        return charge_profiles
class IonMobilitySeparation(Device):
    def __init__(self, name:str = "IonMobilityDevice"):
        super().__init__(name)
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
    def apex_model(self, model: IonMobilityApexModel):
        self._apex_model = model

    @property
    def profile_model(self):
        return self._profile_model

    @profile_model.setter
    def profile_model(self, model: IonMobilityProfileModel):
        self._profile_model = model

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

    @abstractmethod
    def im_to_scan(self):
        pass

class TrappedIon(IonMobilitySeparation):

    def __init__(self, name:str = "TrappedIonMobilitySeparation"):
        super().__init__()

    def run(self, sample: ProteomicsExperimentSampleSlice):
        # scan apex simulation
        scan_apex = self._apex_model.simulate(sample, self)
        # in irt and rt
        sample.add_simulation("simulated_scan_apex", scan_apex)

        # scan profile simulation
        scan_profile = self._profile_model.simulate(sample, self)
        sample.add_simulation("simulated_scan_profile", scan_profile)

    def im_to_scan(self, inv_mob, slope=-880.57513791, intercept=1454.29035506):
        return np.round(inv_mob * slope + intercept).astype(np.int16)


class IonMobilityApexModel(Model):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: IonMobilitySeparation) -> NDArray[np.float64]:
        return super().simulate(sample, device)

class IonMobilityProfileModel(Model):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: IonMobilitySeparation) -> NDArray:
        return super().simulate(sample, device)

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

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: IonMobilitySeparation) -> NDArray:
        data = sample.ions

        ds = self.sequences_tf_dataset(data['mz'], data['charge'], data['sequence'])

        mz = data['mz'].values

        print('predicting mobilities...')
        ccs, _ = self.model.predict(ds)
        one_over_k0 = ccs_to_one_over_reduced_mobility(np.squeeze(ccs), mz, data['charge'].values)

        scans = device.im_to_scan(one_over_k0)
        return scans

class NormalIonMobilityProfileModel(IonMobilityProfileModel):
    def __init__(self):
        super().__init__()

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: IonMobilitySeparation) -> List[ScanProfile]:
        mus = sample.ions["simulated_scan_apex"].values
        ϵ = device.scan_time
        profile_list = []
        for μ in mus:
            σ = 1 # must be sampled

            model_params = { "sigma":σ,
                             "mu":μ,
                             "name":"NORMAL"
                            }

            normal = norm(loc=μ, scale=σ)
            # start and end value (in retention times)
            s_rt, e_rt = normal.ppf([0.05,0.95])
            # as frames
            s_scan, e_scan = int(s_rt), int(e_rt)

            profile_scans = np.arange(s_scan,e_scan+1)
            profile_middle_times = profile_scans + 0.5
            profile_rel_intensities = normal.pdf(profile_middle_times)* ϵ

            profile_list.append(ScanProfile(profile_scans,profile_rel_intensities,model_params))
        return profile_list


class MzSeparation(Device):
    def __init__(self, name:str = "MassSpectrometer"):
        super().__init__(name)
        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: MzSeparationModel):
        self._model = model

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

class TOF(MzSeparation):
    def __init__(self, name:str = "TimeOfFlightMassSpectrometer"):
        super().__init__(name)

    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

class MzSeparationModel(Model):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: MzSeparation):
        return super().simulate(sample, device)

class TOFModel(MzSeparationModel):
    def __init__(self):
        super().__init__()

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: MzSeparation):
        return super().simulate(sample, device)