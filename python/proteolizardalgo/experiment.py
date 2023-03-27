import os
from multiprocessing import Pool
import functools
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

    @abstractmethod
    def assemble(self):
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

        self.assemble()
    @staticmethod
    def _assemble_frame_range(frame_range_start, frame_range_end, ions_in_split, scan_id_min, scan_id_max, default_abundance, resolution):



        # skip if no peptides in split
        if ions_in_split.shape[0] == 0:
            return {}

        ions_in_split.loc[:,"simulated_mz_spectrum"] = ions_in_split["simulated_mz_spectrum"].transform(lambda s: MzSpectrum.from_jsons(jsons=s))

        # construct signal data set
        signal = {f_id:{s_id:[] for s_id in range(scan_id_min, scan_id_max +1)} for f_id in range(frame_range_start, frame_range_end) }

        for _,row in ions_in_split.iterrows():

            ion_frame_start = max(frame_range_start, row["frame_min"])
            ion_frame_end = min(frame_range_end-1, row["frame_max"]) # -1 here because frame_range_end is covered by next frame range

            ion_scan_start = max(scan_id_min, row["scan_min"])
            ion_scan_end = min(scan_id_max, row["scan_max"])

            ion_frame_profile = row["simulated_frame_profile"]
            ion_scan_profile = row["simulated_scan_profile"]

            ion_charge_abundance = row["abundancy"]*row["relative_abundancy"]

            ion_spectrum = row["simulated_mz_spectrum"]

            # frame start and end inclusive
            for f_id in range(ion_frame_start, ion_frame_end+1):
                # scan start and end inclusive
                for s_id in range(ion_scan_start, ion_scan_end+1):

                    abundance = ion_charge_abundance*ion_frame_profile[f_id]*ion_scan_profile[s_id]
                    rel_to_default_abundance = abundance/default_abundance

                    signal[f_id][s_id].append([ion_spectrum,rel_to_default_abundance])

        output_buffer = {}

        for (f_id,frame_dict) in signal.items():
            frame_signal = {}
            for (s_id,spectra_list) in frame_dict.items():
                if spectra_list == []:
                    continue
                frame_signal[s_id] = MzSpectrum(None,f_id, s_id, [], [])
                for (s,r_a) in spectra_list:
                    frame_signal[s_id] += s*r_a
                if frame_signal[s_id].is_empty():
                    del frame_signal[s_id]
                else:
                    frame_signal[s_id] = str(frame_signal[s_id].to_resolution(resolution).to_centroided(1, 1/np.power(10,(resolution-1)) ))
            output_buffer[f_id] = frame_signal

        return output_buffer


    def assemble(self, frame_chunk_size = 120, num_processes = 2):

        with open(self.output_file, "w") as output:
            output.write("{\n")
                # typically for timstof start with 1 and end with 918
        scan_id_min = self.ion_mobility_separation_method.scan_id_min
        scan_id_max = self.ion_mobility_separation_method.scan_id_max
        default_abundance = self.mz_separation_method.model.default_abundance
        resolution = self.mz_separation_method.resolution

        assemble_frame_range = functools.partial(self._assemble_frame_range, scan_id_min = scan_id_min, scan_id_max = scan_id_max, default_abundance = default_abundance, resolution = resolution)

        for f_r in tqdm(range(np.ceil(self.lc_method.num_frames/frame_chunk_size).astype(int))):
            frame_range_start = f_r*frame_chunk_size
            frame_range_end = frame_range_start+frame_chunk_size
            ions_in_frames = self.database.load_frames((frame_range_start, frame_range_end), spectra_as_jsons = True)

            split_positions = np.linspace(frame_range_start, frame_range_end, num= num_processes+1).astype(int)
            split_start = split_positions[:num_processes]
            split_end = split_positions[1:]
            split_data = [ions_in_frames.loc[lambda x: (x["frame_min"] < f_split_max) & (x["frame_max"] >= f_split_min)] for (f_split_min, f_split_max) in zip(split_start,split_end)]

            if num_processes > 1:
                with Pool(num_processes) as pool:
                    results = pool.starmap(assemble_frame_range,   zip(split_start, split_end, split_data) )

            else:
                results = [assemble_frame_range(split_start[0], split_end[0], split_data[0])]

            with open(self.output_file, "a") as output:
                for output_buffer in results:
                    for frame, frame_signal in output_buffer.items():
                        output.write(f"{frame}: {frame_signal} , \n")

        with open(self.output_file, "a") as output:
            output.write("\n}")

