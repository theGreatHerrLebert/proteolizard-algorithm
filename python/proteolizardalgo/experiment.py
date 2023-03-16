import os
from abc import ABC, abstractmethod
import pandas as pd
from tqdm import tqdm

from proteolizarddata.data import MzSpectrum, TimsFrame
from proteolizardalgo.proteome import PeptideDigest, ProteomicsExperimentSampleSlice, ProteomicsExperimentDatabaseHandle
from proteolizardalgo.isotopes import AveragineGenerator
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

        # typically for timstof start with 1 and end with 918
        scan_id_min = self.ion_mobility_separation_method.scan_id_min
        scan_id_max = self.ion_mobility_separation_method.scan_id_max
        scan_id_list = range(scan_id_min,scan_id_max+1)

        avg = AveragineGenerator()
        # construct signal data set
        signal = {}
        spectra_cache = {}
        for f_id in tqdm(range(self.lc_method.num_frames)):
            # for every frame
            # load all ions in that frame
            #empty_frame = TimsFrame(None, f_id, self.lc_method.frame_time_middle(f_id), [],[],[],[],[])

            signal_in_frame = {}
            peptides_in_frame = self.database.load_frame(f_id)
            if peptides_in_frame.shape[0] == 0:
                continue
            # put ions in scan if they appear in that scan
            frame_list = [[] for scan_id in scan_id_list]
            for idx,row in peptides_in_frame.iterrows():

                sequence = row["sequence"]
                charge = row["charge"]
                if sequence in spectra_cache:
                    if charge not in spectra_cache[sequence]:
                        spectra_cache[sequence][charge] = None
                else:
                    spectra_cache[sequence] = {}
                    spectra_cache[sequence][charge] = None

                scan = row["scan_min"]
                while scan <= row["scan_max"]:
                    if scan >= scan_id_min and scan <= scan_id_max:
                        frame_list[scan-scan_id_min].append(idx)
                        scan += 1
                    else:
                        break

            for idx,ion_list in enumerate(frame_list):
                if len(ion_list) == 0:
                    continue
                scan_id = idx+scan_id_min
                scan_spectrum = MzSpectrum(None,f_id,scan_id,[],[])
                # for every scan
                for ion in ion_list:
                    ion_data = peptides_in_frame.loc[ion,:]
                    charge = ion_data["charge"]
                    mass = ion_data["mass_theoretical"]
                    abundancy = ion_data["abundancy"]*ion_data["relative_abundancy"]
                    rel_frame_abundancy = ion_data["simulated_frame_profile"][f_id]
                    rel_scan_abundancy = ion_data["simulated_scan_profile"][scan_id]
                    abundancy *= rel_frame_abundancy*rel_scan_abundancy
                    sequence = ion_data["sequence"]
                    if spectra_cache[sequence][charge] is None:
                        ion_spectrum = avg.generate_spectrum(mass,charge,f_id,scan_id,centroided=False)
                        spectra_cache[sequence][charge] = ion_spectrum
                    else:
                        ion_spectrum = spectra_cache[sequence][charge]
                    default_abundancy = avg.default_abundancy
                    scan_spectrum += abundancy/default_abundancy*ion_spectrum
                signal_in_frame[scan_id] = scan_spectrum.to_resolution(3)
            signal[f_id] = signal_in_frame
        return signal
