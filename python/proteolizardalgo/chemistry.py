import numpy as np

AMINO_ACIDS = {'Lysine': 'K', 'Alanine': 'A', 'Glycine': 'G', 'Valine': 'V', 'Tyrosine': 'Y',
               'Arginine': 'R', 'Glutamic Acid': 'E', 'Phenylalanine': 'F', 'Tryptophan': 'W',
               'Leucine': 'L', 'Threonine': 'T', 'Cysteine': 'C', 'Serine': 'S', 'Glutamine': 'Q',
               'Methionine': 'M', 'Isoleucine': 'I', 'Asparagine': 'N', 'Proline': 'P', 'Histidine': 'H',
               'Aspartic Acid': 'D'}

MASS_PROTON = 1.007276466583

MASS_WATER = 18.010564684

AA_MASSES = {'A': 71.03711, 'C': 103.00919, 'D': 115.02694, 'E': 129.04259, 'F': 147.06841, 'G': 57.02146,
             'H': 137.05891, 'I': 113.08406, 'K': 128.09496, 'L': 113.08406, 'M': 131.04049, 'N': 114.04293,
             'P': 97.05276, 'Q': 128.05858, 'R': 156.10111, 'S': 87.03203, 'T': 101.04768, 'V': 99.06841,
             'W': 186.07931, 'Y': 163.06333, '<AC>': 42.010565, '<OX>': 15.994915, 'U': 168.964203,
             '<CM>': 57.021464, '<PH>': 79.966331, '<CY>': 0.0, '<START>': 0.0, '<END>': 0.0}

VARIANT_DICT = {'L': ['L'], 'E': ['E'], 'S': ['S', 'S-<PH>'], 'A': ['A'], 'V': ['V'], 'D': ['D'], 'G': ['G'],
                '<END>': ['<END>'], 'P': ['P'], '<START>': ['<START>', '<START>-<AC>'], 'T': ['T', 'T-<PH>'],
                'I': ['I'], 'Q': ['Q'], 'K': ['K', 'K-<AC>'], 'N': ['N'], 'R': ['R'], 'F': ['F'], 'H': ['H'],
                'Y': ['Y', 'Y-<PH>'], 'M': ['M', 'M-<OX>'],
                'W': ['W'], 'C': ['C', 'C-<CY>', 'C-<CM>'], 'C-<CM>': ['C', 'C-<CY>', 'C-<CM>']}


def get_mono_isotopic_weight(sequence_tokenized: list[str]) -> float:
    flat_seq = [char for sublist in [c.split('-') for c in sequence_tokenized] for char in sublist]
    return sum(map(lambda c: AA_MASSES[c], flat_seq)) + MASS_WATER


def get_mass_over_charge(mass: float, charge: int) -> float:
    return (mass / charge) + MASS_PROTON


def reduced_mobility_to_ccs(one_over_k0, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    convert reduced ion mobility (1/k0) to CCS
    :param one_over_k0: reduced ion mobility
    :param charge: charge state of the ion
    :param mz: mass-over-charge of the ion
    :param mass_gas: mass of drift gas
    :param temp: temperature of the drift gas in C°
    :param t_diff: factor to translate from C° to K
    """
    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return (SUMMARY_CONSTANT * charge) / (np.sqrt(reduced_mass * (temp + t_diff)) * 1 / one_over_k0)


def ccs_to_one_over_reduced_mobility(ccs, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    convert CCS to 1 over reduced ion mobility (1/k0)
    :param ccs: collision cross-section
    :param charge: charge state of the ion
    :param mz: mass-over-charge of the ion
    :param mass_gas: mass of drift gas (N2)
    :param temp: temperature of the drift gas in C°
    :param t_diff: factor to translate from C° to K
    """
    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return  ((np.sqrt(reduced_mass * (temp + t_diff))) * ccs) / (SUMMARY_CONSTANT * charge)



