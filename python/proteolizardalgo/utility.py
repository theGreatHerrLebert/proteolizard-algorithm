import numpy as np

from tqdm import tqdm
from proteolizarddata.data import MzSpectrum
from proteolizardalgo.hashing import ReferencePattern

def peak_width_preserving_mz_transform(
        mz: np.array,
        M0: float = 500,
        resolution: float = 50_000):
    """
    Transform values into an index that fixes the width of the peak at half full width.
    Arguments:
        mz (np.array): An array of mass to charge ratios to transform.
        M0 (float): The m/z value at which TOFs resolution is reported.
        resolution (float): The resolution of the TOF instrument.
    """
    return (np.log(mz) - np.log(M0)) / np.log1p(1 / resolution)


def bins_to_mz(mz_bin, win_len):
    return np.abs(mz_bin) * win_len + (int(mz_bin < 0) * (0.5 * win_len))


def get_ref_pattern_as_spectra(ref_patterns):
    """

    """

    spec_list = []

    for index, row in ref_patterns.iterrows():

        if 150 <= (row['m'] + 1) / row['z'] <= 1700:
            last_contrib = row['last_contrib']

            pattern = row['pattern'][:1000]

            mz_bin = np.arange(1000)

            both = list(zip(mz_bin, pattern))

            both = list(filter(lambda x: x[1] >= 0.001, both))

            mz = [x[0] for x in both]
            i = [x[1] for x in both]

            first_peak = np.where(np.diff(mz) > 1)[0][0]
            max_i = np.argmax(i[:first_peak])
            mono_mz_max = mz[max_i]

            mz = (mz - mono_mz_max) / 100 + row['m'] / row['z']

            mz = np.round(np.array(mz), 2) + 1.0
            i = np.array([int(x) for x in i])

            spectum = MzSpectrum(None, -1, -1, mz, i)
            spec_list.append((row['m'], row['z'], spectum, last_contrib))

    return spec_list


def get_refspec_list(ref_patterns, win_len=5):
    """

    """

    d_list = []

    for spec in tqdm(ref_patterns, desc='creating reference patterns', ncols=100):
        m = spec[0]
        z = spec[1]
        sp = spec[2]
        lc = spec[3]
        peak_width = 0.04
        mz_mono = np.round((m + 1.001) / z, 2) + peak_width
        mz_bin = np.floor(mz_mono / win_len)
        ref_spec = ReferencePattern(mz_bin=mz_bin, mz_mono=mz_mono, charge=z, mz_last_contrib=lc,
                                    spectrum=sp, window_length=win_len)

        d_list.append((mz_bin, ref_spec))

    return d_list


def create_reference_dict(D):
    # create members
    keys = set([t[0] for t in D])
    ref_d = dict([(k, []) for k in keys])

    for b, p in D:
        ref_d[b].append(p)

    tmp_dict = {}
    # go over all keys
    for key, values in ref_d.items():

        # create charges
        r = np.arange(5) + 1

        # go over all charge states
        for c in r:

            # go over all reference pattern
            for ca in values:

                # append exactly one charge state per reference bin
                if c == ca.charge:
                    # if key already exists, append
                    if key in tmp_dict:
                        tmp_dict[key].append(ca)
                        break

                    # if key does not exist, create new list
                    else:
                        tmp_dict[key] = [ca]
                        break

    return tmp_dict
