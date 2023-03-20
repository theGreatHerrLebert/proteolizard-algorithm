from __future__ import annotations
import os
import warnings
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import sqlite3
from proteolizardalgo.feature import RTProfile, ScanProfile, ChargeProfile
from proteolizardalgo.utility import preprocess_max_quant_sequence, TokenSequence
from proteolizardalgo.chemistry import get_mono_isotopic_weight, MASS_PROTON
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, List, Union

class ENZYME(Enum):
    TRYPSIN = 1


class ORGANISM(Enum):
    HOMO_SAPIENS = 1


class Enzyme(ABC):
    def __init__(self, name: ENZYME):
        self.name = name

    @abstractmethod
    def calculate_cleavages(self, sequence: str) -> np.array:
        pass

    def digest(self, sequence: str, missed_cleavages: int, min_length) -> list[str]:
        pass


class Trypsin(Enzyme):
    def __init__(self, name: ENZYME = ENZYME.TRYPSIN):
        super().__init__(name)
        self.name = name

    def calculate_cleavages(self, sequence):

        cut_sites = [0]

        for i in range(len(sequence) - 1):
            # cut after every 'K' or 'R' that is not followed by 'P'
            if ((sequence[i] == 'K') or (sequence[i] == 'R')) and sequence[i + 1] != 'P':
                cut_sites += [i + 1]

        cut_sites += [len(sequence)]

        return np.array(cut_sites)

    def __repr__(self):
        return f'Enzyme(name: {self.name.name})'

    def digest(self, sequence, abundancy, missed_cleavages=0, min_length=7):
        assert 0 <= missed_cleavages <= 2, f'Number of missed cleavages might be between 0 and 2, was: {missed_cleavages}'

        cut_sites = self.calculate_cleavages(sequence)
        pairs = np.c_[cut_sites[:-1], cut_sites[1:]]

        # TODO: implement
        if missed_cleavages == 1:
            pass
        if missed_cleavages == 2:
            pass

        dig_list = []

        for s, e in pairs:
            dig_list += [sequence[s: e]]

        # pair sequence digests with indices
        wi = zip(dig_list, pairs)
        # filter out short sequences and clean index display
        wi = map(lambda p: (p[0], p[1][0], p[1][1]), filter(lambda s: len(s[0]) >= min_length, wi))

        return list(map(lambda e: {'sequence': e[0], 'start': e[1], 'end': e[2], 'abundancy': abundancy}, wi))


class PeptideDigest:
    def __init__(self, data: pd.DataFrame, organism: ORGANISM, enzyme: ENZYME):
        self.data = data
        self.organism = organism
        self.enzyme = enzyme

    def __repr__(self):
        return f'PeptideMix(Sample: {self.organism.name}, Enzyme: {self.enzyme.name}, number peptides: {self.data.shape[0]})'


class ProteinSample:

    def __init__(self, data: pd.DataFrame, name: ORGANISM):
        self.data = data
        self.name = name

    def digest(self, enzyme: Enzyme, missed_cleavages: int = 0, min_length: int = 7) -> PeptideDigest:

        digest = self.data.apply(lambda r: enzyme.digest(r['sequence'], r['abundancy'], missed_cleavages, min_length), axis=1)

        V = zip(self.data['id'].values, digest.values)

        r_list = []

        for (gene, peptides) in V:
            for (pep_idx, pep) in enumerate(peptides):
                if pep['sequence'].find('X') == -1:
                    pep['gene_id'] = gene
                    pep['pep_id'] = f"{gene}_{pep_idx}"
                    pep['sequence'] = '_' + pep['sequence'] + '_'
                    pep['sequence_tokenized'] = preprocess_max_quant_sequence(pep['sequence'])
                    pep['mass_theoretical'] = get_mono_isotopic_weight(pep['sequence_tokenized'])
                    pep['sequence_tokenized'] = TokenSequence(pep['sequence_tokenized']).jsons
                    r_list.append(pep)

        return PeptideDigest(pd.DataFrame(r_list), self.name, enzyme.name)

    def __repr__(self):
        return f'ProteinSample(Organism: {self.name.name})'

class ProteomicsExperimentDatabaseHandle:
    def __init__(self,path:str):
        if os.path.exists(path):
            warnings.warn("Database exists")
        self.con = sqlite3.connect(path)
        self._chunk_size = None

    def push(self, table_name:str, data:PeptideDigest):
        if table_name == "PeptideDigest":
            assert isinstance(data, PeptideDigest), "For pushing to table 'PeptideDigest' data type must be `PeptideDigest`"
            df = data.data
        else:
            raise ValueError("This Table does not exist and is not supported")

        df.to_sql(table_name, self.con, if_exists="replace")

    def update(self, data_slice: ProteomicsExperimentSampleSlice):
        assert isinstance(data_slice, ProteomicsExperimentSampleSlice)
        df_separated_peptides = data_slice.peptides.apply(self.make_sql_compatible)
        df_ions = data_slice.ions.apply(self.make_sql_compatible)
        df_separated_peptides.to_sql("SeparatedPeptides", self.con, if_exists="append")
        df_ions.to_sql("Ions", self.con , if_exists="append")

    def load(self, table_name:str, query:Optional[str] = None):
        if query is None:
            query = f"SELECT * FROM {table_name}"
        return pd.read_sql(query,self.con, index_col="index")

    def load_chunks(self, chunk_size: int, query:Optional[str] = None):
        if query is None:
            query = "SELECT * FROM PeptideDigest"
        self.__chunk_generator =  pd.read_sql(query,self.con, chunksize=chunk_size, index_col="index")
        for chunk in self.__chunk_generator:
            yield(ProteomicsExperimentSampleSlice(peptides = chunk))

    def load_frame(self, frame_id:int):
        query = (
                "SELECT SeparatedPeptides.pep_id, "
                "SeparatedPeptides.sequence, "
                "SeparatedPeptides.simulated_frame_profile, "
                "SeparatedPeptides.mass_theoretical, "
                "SeparatedPeptides.abundancy, "
                "Ions.mz, "
                "Ions.charge, "
                "Ions.relative_abundancy, "
                "Ions.scan_min, "
                "Ions.scan_max, "
                "Ions.simulated_scan_profile "
                "FROM SeparatedPeptides "
                "INNER JOIN Ions "
                "ON SeparatedPeptides.pep_id = Ions.pep_id "
                f"AND SeparatedPeptides.frame_min <= {frame_id} "
                f"AND SeparatedPeptides.frame_max >= {frame_id} "
                )
        df = pd.read_sql(query, self.con)

        # unzip jsons
        df.loc[:,"simulated_scan_profile"] = df["simulated_scan_profile"].transform(lambda sp: ScanProfile(jsons=sp))
        df.loc[:,"simulated_frame_profile"] = df["simulated_frame_profile"].transform(lambda rp: RTProfile(jsons=rp))

        return df

    @staticmethod
    def make_sql_compatible(column):
        if column.size < 1:
            return
        if isinstance(column.iloc[0], (RTProfile,ScanProfile,ChargeProfile)):
            return column.apply(lambda x: x.jsons)
        else:
            return column

class ProteomicsExperimentSampleSlice:
    """
    exposed dataframe of database
    """
    def __init__(self, peptides:pd.DataFrame, ions:Optional[pd.DataFrame]=None):
        self.peptides = peptides
        self.ions = ions

    def add_simulation(self, simulation_name:str, simulation_data: ArrayLike):
        accepted_peptide_simulations = [
                                "simulated_irt_apex",
                                "simulated_frame_apex",
                                "simulated_frame_profile",
                                "simulated_charge_profile",
                                ]

        accepted_ion_simulations = [
                                "simulated_scan_apex",
                                "simulated_one_over_k0",
                                "simulated_scan_profile",
                                ]

        # for profiles store min and max values
        get_min_position = np.vectorize(lambda p:p.positions.min(), otypes=[int])
        get_max_position = np.vectorize(lambda p:p.positions.max(), otypes=[int])

        if simulation_name == "simulated_frame_profile":

            self.peptides["frame_min"] = get_min_position(simulation_data)
            self.peptides["frame_max"] = get_max_position(simulation_data)

        elif simulation_name == "simulated_charge_profile":
            ions_dict = {
                "sequence":[],
                "pep_id":[],
                "mz":[],
                "charge":[],
                "relative_abundancy":[]
                }
            sequences = self.peptides["sequence"].values
            pep_ids = self.peptides["pep_id"].values
            masses = self.peptides["mass_theoretical"].values

            for (s, pi, m, charge_profile) in zip(sequences, pep_ids, masses, simulation_data):
                for (c, r) in charge_profile:
                    ions_dict["sequence"].append(s)
                    ions_dict["pep_id"].append(pi)
                    ions_dict["charge"].append(c)
                    ions_dict["relative_abundancy"].append(r)
                    ions_dict["mz"].append(m/c + MASS_PROTON)
            self.ions = pd.DataFrame(ions_dict)

            self.peptides["charge_min"] = get_min_position(simulation_data)
            self.peptides["charge_max"] = get_max_position(simulation_data)

        elif simulation_name == "simulated_scan_profile":

            self.ions["scan_min"] = get_min_position(simulation_data)
            self.ions["scan_max"] = get_max_position(simulation_data)

        if simulation_name in accepted_peptide_simulations:
            self.peptides[simulation_name] = simulation_data

        elif simulation_name in accepted_ion_simulations:
            self.ions[simulation_name] = simulation_data

        else:
            raise ValueError(f"Simulation name '{simulation_name}' is not defined")