import pandas as pd
import numpy as np


def preprocess_max_quant_evidence(exp: pd.DataFrame) -> pd.DataFrame:
    """
    select columns from evidence txt, rename to ionmob naming convention and transform to raw data rt in seconds
    Args:
        exp: a MaxQuant evidence dataframe from evidence.txt table
    Returns: cleaned evidence dataframe, columns renamed to ionmob naming convention
    """

    # select columns
    exp = exp[['m/z', 'Mass', 'Charge', 'Modified sequence', 'Retention time',
               'Retention length', 'Ion mobility index', 'Ion mobility length', '1/K0', '1/K0 length',
               'Number of isotopic peaks', 'Max intensity m/z 0', 'Intensity', 'Raw file', 'CCS']].rename(
        # rename columns to ionmob naming convention
        columns={'m/z': 'mz', 'Mass': 'mass',
                 'Charge': 'charge', 'Modified sequence': 'sequence', 'Retention time': 'rt',
                 'Retention length': 'rt_length', 'Ion mobility index': 'im', 'Ion mobility length': 'im_length',
                 '1/K0': 'inv_ion_mob', '1/K0 length': 'inv_ion_mob_length', 'CCS': 'ccs',
                 'Number of isotopic peaks': 'num_peaks', 'Max intensity m/z 0': 'mz_max_intensity',
                 'Intensity': 'intensity', 'Raw file': 'raw'}).dropna()

    # transform retention time from minutes to seconds as stored in tdf raw data
    exp['rt'] = exp.apply(lambda r: r['rt'] * 60, axis=1)
    exp['rt_length'] = exp.apply(lambda r: r['rt_length'] * 60, axis=1)
    exp['rt_start'] = exp.apply(lambda r: r['rt'] - r['rt_length'] / 2, axis=1)
    exp['rt_stop'] = exp.apply(lambda r: r['rt'] + r['rt_length'] / 2, axis=1)

    exp['im_start'] = exp.apply(lambda r: int(np.round(r['im'] - r['im_length'] / 2)), axis=1)
    exp['im_stop'] = exp.apply(lambda r: int(np.round(r['im'] + r['im_length'] / 2)), axis=1)

    exp['inv_ion_mob_start'] = exp.apply(lambda r: r['inv_ion_mob'] - r['inv_ion_mob_length'] / 2, axis=1)
    exp['inv_ion_mob_stop'] = exp.apply(lambda r: r['inv_ion_mob'] + r['inv_ion_mob_length'] / 2, axis=1)

    # remove duplicates
    exp = exp.drop_duplicates(['sequence', 'charge', 'rt', 'ccs'])

    return exp


def preprocess_max_quant_sequence(s, old_annotation=False):
    """
    :param s:
    :param old_annotation:
    """

    seq = s[1:-1]

    is_acc = False

    if old_annotation:
        seq = seq.replace('(ox)', '$')

        if seq.find('(ac)') != -1:
            is_acc = True
            seq = seq.replace('(ac)', '')

    else:
        seq = seq.replace('(Oxidation (M))', '$')
        seq = seq.replace('(Phospho (STY))', '&')

        if seq.find('(Acetyl (Protein N-term))') != -1:
            is_acc = True
            seq = seq.replace('(Acetyl (Protein N-term))', '')

    # form list from string
    slist = list(seq)

    tmp_list = []

    for item in slist:
        if item == '$':
            tmp_list.append('<OX>')

        elif item == '&':
            tmp_list.append('<PH>')

        else:
            tmp_list.append(item)

    slist = tmp_list

    r_list = []

    for i, char in enumerate(slist):

        if char == '<OX>':
            C = slist[i - 1]
            C = C + '-<OX>'
            r_list = r_list[:-1]
            r_list.append(C)

        elif char == 'C':
            r_list.append('C-<CM>')

        elif char == '<PH>':
            C = slist[i - 1]
            C = C + '-<PH>'
            r_list = r_list[:-1]
            r_list.append(C)

        else:
            r_list.append(char)

    if is_acc:
        return ['<START>-<AC>'] + r_list + ['<END>']

    return ['<START>'] + r_list + ['<END>']