import os
import json
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
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
        self.output_file = f"{os.path.dirname(path)}/output.json"
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


class LcImsMsMs(ProteomicsExperiment):
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
            self.mz_separation_method.run(data_chunk)
            self.database.update(data_chunk)

        # typically for timstof start with 1 and end with 918
        scan_id_min = self.ion_mobility_separation_method.scan_id_min
        scan_id_max = self.ion_mobility_separation_method.scan_id_max

        # construct signal data set
        signal = {f_id:{s_id:[] for s_id in range(scan_id_min, scan_id_max +1)} for f_id in range(self.lc_method.num_frames) }
        frame_chunk_size = 100
        for f_r in tqdm(range(np.ceil(self.lc_method.num_frames/frame_chunk_size).astype(int))):

            # load all ions in that frame range
            frame_range_start = f_r * frame_chunk_size
            frame_range_end = frame_range_start + frame_chunk_size
            peptides_in_frames = self.database.load_frames((frame_range_start, frame_range_end))

            # skip if no peptides in frame
            if peptides_in_frames.shape[0] == 0:
                continue

            for _,row in peptides_in_frames.iterrows():

                frame_start = max(frame_range_start, row["frame_min"])
                frame_end = min(frame_range_end-1, row["frame_max"]) # -1 here because frame_range_end is covered by next frame range

                scan_start = max(scan_id_min, row["scan_min"])
                scan_end = min(scan_id_max, row["scan_max"])

                frame_profile = row["simulated_frame_profile"]
                scan_profile = row["simulated_scan_profile"]

                charge_abundance = row["abundancy"]*row["relative_abundancy"]

                spectrum = row["simulated_mz_spectrum"]

                # frame start and end inclusive
                for f_id in range(frame_start, frame_end+1):
                    # scan start and end inclusive
                    for s_id in range(scan_start, scan_end+1):

                        abundance = charge_abundance*frame_profile[f_id]*scan_profile[s_id]
                        rel_to_default_abundance = abundance/self.mz_separation_method.model.default_abundance

                        signal[f_id][s_id].append([spectrum,rel_to_default_abundance])
        with open(self.output_file, "w") as output:
                output.write("{\n")

        output_buffer = {}
        for (f_id,frame_dict) in tqdm(signal.items()):
            frame_signal = {s_id:MzSpectrum(None, f_id, s_id,[],[]) for s_id in range(scan_id_min, scan_id_max +1)}
            for (s_id,spectra_list) in frame_dict.items():
                for (s,r_a) in spectra_list:
                    frame_signal[s_id] += s*r_a

                if frame_signal[s_id].sum_intensity() <= 0:
                    del frame_signal[s_id]
                else:
                    frame_signal[s_id] = frame_signal[s_id].to_resolution(self.mz_separation_method.resolution).to_jsons(only_spectrum=True)
            output_buffer[f_id] = frame_signal

            if (f_id+1) % 500 == 1:
                continue
                with open(self.output_file, "a") as output:
                    for frame, frame_signal in output_buffer.items():
                        output.write(f"{frame}: {json.dumps(frame_signal)} , \n")
                output_buffer.clear()
        with open(self.output_file, "a") as output:
                output.write("\n}")