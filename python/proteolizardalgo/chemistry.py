import numpy as np
import mendeleev as me

AMINO_ACIDS = {'Lysine': 'K', 'Alanine': 'A', 'Glycine': 'G', 'Valine': 'V', 'Tyrosine': 'Y',
               'Arginine': 'R', 'Glutamic Acid': 'E', 'Phenylalanine': 'F', 'Tryptophan': 'W',
               'Leucine': 'L', 'Threonine': 'T', 'Cysteine': 'C', 'Serine': 'S', 'Glutamine': 'Q',
               'Methionine': 'M', 'Isoleucine': 'I', 'Asparagine': 'N', 'Proline': 'P', 'Histidine': 'H',
               'Aspartic Acid': 'D'}

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

MASS_PROTON = 1.007276466583

MASS_WATER = 18.010564684
# IUPAC standard in Kelvin
STANDARD_TEMPERATURE = 273.15
# IUPAC standard in Pa
STANDARD_PRESSURE = 1e5
# IUPAC elementary charge
ELEMENTARY_CHARGE = 1.602176634e-19
# IUPAC BOLTZMANN'S CONSTANT
K_BOLTZMANN = 1.380649e-23


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


class ChemicalCompound:

    def _calculate_molecular_mass(self):
        mass = 0
        for (atom, abundance) in self.element_composition.items():
            mass += me.element(atom).atomic_weight * abundance
        return mass

    def __init__(self, formula):
        self.element_composition = self.get_composition(formula)
        self.mass = self._calculate_molecular_mass()

    def get_composition(self, formula:str):
        """
        Parse chemical formula into Dict[str:int] with
        atoms as keys and the respective counts as values.

        :param formula: Chemical formula of compound e.g. 'C6H12O6'
        :type formula: str
        :return: Dictionary Atom: Count
        :rtype: Dict[str:int]
        """
        if formula.startswith("("):
            assert formula.endswith(")")
            formula = formula[1:-1]

        tmp_group = ""
        tmp_group_count = ""
        depth = 0
        comp_list = []
        comp_counts = []

        # extract components: everything in brackets and atoms
        # extract component counts: number behind component or 1
        for (i,e) in enumerate(formula):
            if e == "(":
                depth += 1
                if depth == 1:
                    if tmp_group != "":
                        comp_list.append(tmp_group)
                        tmp_group = ""
                        if tmp_group_count == "":
                            comp_counts.append(1)
                        else:
                            comp_counts.append(int(tmp_group_count))
                            tmp_group_count = ""
                tmp_group += e
                continue
            if e == ")":
                depth -= 1
                tmp_group += e
                continue
            if depth > 0:
                tmp_group += e
                continue
            if e.isupper():
                if tmp_group != "":
                    comp_list.append(tmp_group)
                    tmp_group = ""
                    if tmp_group_count == "":
                        comp_counts.append(1)
                    else:
                        comp_counts.append(int(tmp_group_count))
                        tmp_group_count = ""
                tmp_group += e
                continue
            if e.islower():
                tmp_group += e
                continue
            if e.isnumeric():
                tmp_group_count += e
        if tmp_group != "":
            comp_list.append(tmp_group)
            if tmp_group_count == "":
                comp_counts.append(1)
            else:
                comp_counts.append(int(tmp_group_count))

        # assemble dictionary from component lists
        atom_dict = {}
        for (comp,count) in zip(comp_list,comp_counts):
            if not comp.startswith("("):
                atom_dict[comp] = count
            else:
                atom_dicts_depth = self.get_composition(comp)
                for atom in atom_dicts_depth:
                    atom_dicts_depth[atom] *= count
                    if atom in atom_dict:
                        atom_dict[atom] += atom_dicts_depth[atom]
                    else:
                        atom_dict[atom] = atom_dicts_depth[atom]
                atom_dicts_depth = {}
        return atom_dict

class BufferGas(ChemicalCompound):

    def __init__(self, formula: str, density: float):
        super().__init__(formula)
        self.N0 = density

