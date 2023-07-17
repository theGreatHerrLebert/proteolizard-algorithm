
import os
import sys
import warnings
from typing import Dict
import tqdm
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import pyopenms
from proteolizardalgo.chemistry import get_mono_isotopic_weight
from proteolizarddata.data import PyTimsDataHandleDDA
from proteolizardalgo.utility import tokenize_proforma_sequence, max_quant_to_proforma

def get_isotope_intensities(sequence: str, generator: pyopenms.CoarseIsotopePatternGenerator):
    """
    Gets hypothetical intensities of isotopic peaks.

    :param sequence: Peptide sequence in proForma format.
    :type sequence: str
    :param generator: pyopenms generator for isotopic pattern calculation
    :type generator: pyopenms.CoarseIsotopePatternGenerator
    :return: List of isotopic peak intensities
    :rtype: List
    """
    aa_sequence = sequence.strip("_")
    peptide= pyopenms.AASequence().fromString(aa_sequence.replace("[","(").replace("]",")"))
    formula = peptide.getFormula()
    distribution = generator.run(formula)
    intensities = [i.getIntensity() for i in distribution.getContainer()]
    return intensities

def preprocess_mq_dataframe(evidence_df: pd.DataFrame, data_handle: PyTimsDataHandleDDA, minimum_mz_intensity_fraction: float, margin_mz_low: float, margin_mz_high: float) -> pd.DataFrame:
    """
    Transforms filtered MaxQuant's evidence df for extraction of feature raw data.
    E.g. calculate start and stop for retention-time-, scan- and m/z
    dimension. While start and stop position in retention-time- and scan dimension
    are calculated based on MaxQuant's `Retention length` and `Ion mobility length`, respectively,
    mz start and stop position is based on OpenMS's predicted isotopic peak intensities

    :param evidence_df: Filtered MaxQuant evidence DataFrame
    :type evidence_df: pd.DataFrame
    :param data_handle:  Proteolizard handle of experimental raw data
    :type data_handle: PyTimsDataHandleDDA
    :param minimum_mz_intensity_fraction: Minimum fraction of total intensity to collect in mz dimension.
    :type minimum_mz_intensity_fraction: float
    :param margin_mz_low: Margin for start in mz dimension (minus first peak)
    :type margin_mz_low: float
    :param margin_mz_high: Margin for stop in mz dimension (plus last peak)
    :type margin_mz_high: float
    :return: DataFrame with new columns needed for data extraction
    :rtype: pd.DataFrame
    """
    generator = pyopenms.CoarseIsotopePatternGenerator()
    generator.setRoundMasses(True)
    generator.setMaxIsotope(20)
    seconds_per_minute = 60
    return (
            evidence_df.assign(Sequence = lambda df_: df_["Sequence"].apply(lambda s: max_quant_to_proforma(s)),
                                Sequence_tokenized = lambda df_: df_["Sequence"].apply(lambda s: tokenize_proforma_sequence(s.strip("_"))),
                                Mass_theoretical = lambda df_: df_["Sequence_tokenized"].apply(lambda s: get_mono_isotopic_weight(s)),
                                Mz_experiment = lambda df_: df_["m/z"]+df_["Uncalibrated - Calibrated m/z [Da]"],
                                Isotope_intensities = lambda df_: df_["Sequence"].apply(lambda s: get_isotope_intensities(s, generator)),
                                Num_peaks = lambda df_: df_["Isotope_intensities"].apply(lambda ii: np.argmax(np.cumsum(ii)>=minimum_mz_intensity_fraction)+1),
                                Rt_start = lambda df_: (df_["Retention time"]-df_["Retention length"]/2) * seconds_per_minute,
                                Rt_stop = lambda df_: (df_["Retention time"]+df_["Retention length"]/2) * seconds_per_minute,
                                Retention_time_sec = lambda df_: df_["Retention time"] * seconds_per_minute,
                                Im_index_start = lambda df_: np.floor(df_["Ion mobility index"]-df_["Ion mobility length"]/2).astype(int),
                                Im_index_stop = lambda df_: np.ceil(df_["Ion mobility index"]+df_["Ion mobility length"]/2).astype(int),
                                Mz_start = lambda df_: df_["Mz_experiment"]-margin_mz_low,
                                Mz_stop = lambda df_: df_["Mz_experiment"]+(df_["Num_peaks"]-1)/df_["Charge"]+margin_mz_high,
                                Frame_apex = lambda df_: df_["Retention_time_sec"].apply(lambda _rt: data_handle.rt_to_closest_frame_id(_rt))
                                )
    )

def get_raw_data_maxquant(evidence_df: pd.DataFrame, data_handle: PyTimsDataHandleDDA) -> pd.DataFrame:
    """
    Extracts raw data from `data_handle` for each feature (row) in `evidence_df`.

    :param evidence_df: Filtered and preprocessed MaxQuant evidence DataFrame
    :type evidence_df: pd.DataFrame
    :param data_handle:  Proteolizard handle of experimental raw data
    :type data_handle: PyTimsDataHandleDDA
    :return: DataFrame with feature datapoints
    :rtype: pd.DataFrame
    """
    raw_data = evidence_df.apply(lambda df_: data_handle.get_raw_points((df_["Rt_start"],df_["Rt_stop"]),(df_["Im_index_start"], df_["Im_index_stop"]), (df_["Mz_start"], df_["Mz_stop"]), False), axis = 1)
    i = raw_data.apply(lambda s_: s_.intensity.tolist())
    m = raw_data.apply(lambda s_: s_.mz.tolist())
    s = raw_data.apply(lambda s_: s_.scan.tolist())
    f = raw_data.apply(lambda s_: s_.frame.tolist())

    df = evidence_df.assign(intensities = i, mzs = m, scans = s, frames = f)
    return df.loc[:,["Sequence",
                     "Mass_theoretical",
                     "Charge",
                     "m/z",
                     "Mz_experiment",
                     "Uncalibrated - Calibrated m/z [Da]",
                     "Num_peaks",
                     "Number of isotopic peaks",
                     "PEP",
                     "Match score",
                     "id",
                     "Protein group IDs",
                     "Rt_start",
                     "Rt_stop",
                     "Im_index_start",
                     "Im_index_stop",
                     "Mz_start",
                     "Mz_stop",
                     "Retention time",
                     "Retention_time_sec",
                     "Frame_apex",
                     "Ion mobility index",
                     "intensities",
                     "mzs",
                     "scans",
                     "frames",
                     "Isotope_intensities",
                     "Raw file"]
                  ].rename({
                            "m/z":"Calibrated_mz",
                            "Number of isotopic peaks":"Num_peaks_MQ",
                            "Retention time": "Retention_time_min"
                            })

def load_evidence_table(evidence_path: str) -> pd.DataFrame:
    """
    Loads data of evidence files (all .txt files in `evidence_path`)

    :param evidence_path: Path to MaxQuant evidence files
    :type evidence_path: str
    :return: Combined Dataframe
    :rtype: pd.DataFrame
    """
    #
    evidence_files = [f"{evidence_path}/{file}" for file in os.listdir(evidence_path) if file.endswith(".txt")]
    evidence_dfs = []
    for ef in evidence_files:
        evidence_dfs.append(pd.read_table(ef))
    return pd.concat(evidence_dfs)

def filter_evidence_table(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicates that are not explained by charge-, sequence-, run-, or modification differences
    and removes sequences that are marked to likely be decoy sequences or contaminations by MaxQuant

    :param evidence_df: Evidence dataframe (combined of all evidence.txt files)
    :type evidence_df: pd.DataFrame
    :return: filtered DataFrame
    :rtype: pd.DataFrame
    """
    return evidence_df.drop_duplicates(["Sequence","Charge","Modifications","Raw file"], keep=False).loc[lambda _df: (_df["Reverse"]!="+")&(_df["Potential contaminant"]!="+"), :]

def find_raw_data_folders(d_path: str, files_in_evidence: ArrayLike)-> Dict[str,str]:
    """
    Find all timsTOF raw data folders in `d_path` and warn
    if not all experiments referenced in evidence are found.

    :return: Dictionary with file name as referenced in evidence
             as keys and path to .d folder as values.
    :rtype: Dict
    """
    d_dirs = {folder[:-2]:f"{d_path}/{folder}" for folder in os.listdir(d_path) if folder.endswith(".d")}
    for raw_data_reference in files_in_evidence:
        if raw_data_reference not in d_dirs:
            warnings.warn(f"{raw_data_reference} in evidence table but not in raw data folder")
    return d_dirs

def process_in_chunks(evidence_path: str, d_path: str, output_path: str, params: Dict) -> None:
    """
    Driver function of this script. Finds and loads evidence files in `evidence_path`,
    finds .d experiment folders in `d_path` and stores processed evidence table
    with raw data points in parquet format in `output_path`

    :param evidence_path: Path to folder with MaxQuant evidence.txt files
    :type evidence_path: str
    :param d_path: Path to folder with timsTOF .d experiment folders
    :type d_path: str
    :param output_path: Desired output path.
    :type output_path: str
    :param params: User defined parameters
    :type params: Dict
    """
    # load params
    chunk_size = params["chunk_size"]
    reduced =params["reduced"]
    sample_fraction =params["sample_fraction"]
    margin_mz_low =params["margin_mz_low"]
    margin_mz_high =params["margin_mz_high"]
    min_rel_intensity_mz =params["min_rel_intensity_mz"]

    # find evidence files and filter e.g. duplicants, contminations and decoy sequences
    evidence = filter_evidence_table(load_evidence_table(evidence_path))
    # find raw data folders
    raw_data_folders = find_raw_data_folders(d_path, evidence["Raw file"].unique())


    evidence_grouped_by_experiment = evidence.groupby(by="Raw file")
    for experiment_name, experiment_evidence in evidence_grouped_by_experiment:
        if experiment_name not in raw_data_folders:
            continue
        dh = PyTimsDataHandleDDA(raw_data_folders[experiment_name])

        # split in chunks for raw data extraction
        evidence_chunks = [experiment_evidence.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(len(experiment_evidence)//chunk_size+1)]
        for i,ec in tqdm.tqdm(enumerate(evidence_chunks),total=len(evidence_chunks)):

            if reduced:
                ec = ec.sample(frac=sample_fraction)

            if ec.empty:
                continue
            df = preprocess_mq_dataframe(ec, dh, min_rel_intensity_mz, margin_mz_low, margin_mz_high)
            df_with_raw_data = get_raw_data_maxquant(df, dh)
            df_with_raw_data.to_parquet(f"{output_path}/{experiment_name}_{i}.parquet")

if __name__ == "__main__":
    # Defineable Parameters
    PARAMS = {
            "margin_mz_low" : 0.1,
            "margin_mz_high" : 0.1,
            "min_rel_intensity_mz" : 0.98,
            "chunk_size" : 1000,
            "reduced" : True,
            "sample_fraction" : 0.01
            }
    # Test if all necessary paths were given
    try:
        _, EVIDENCE_PATH, D_PATH, OUTPUT_PATH = sys.argv
    except ValueError:
        raise ValueError("This python script must be given a path to evidence.txt files, experiment.d folders and an output path")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    process_in_chunks(EVIDENCE_PATH, D_PATH, OUTPUT_PATH, PARAMS)




