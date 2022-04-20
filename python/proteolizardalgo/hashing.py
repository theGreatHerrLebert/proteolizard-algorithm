import tensorflow as tf
import numpy as np
import warnings
import libproteolizardalgorithm as pla
import pandas as pd

from proteolizarddata.data import TimsFrame

from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage.interpolation import shift

from tqdm import tqdm


class TimsHasher:
    """
    :param trials: number of keys to be generated
    :param len_trial: bits per key, aka 0 or 1 for every signum created via random projection
    :param seed: a seed to be able to recreate random hashes for a given parameter setting
    :param resolution: decimals kept when binning the mz axis of a given mass spectrum
    :param num_dalton: length of a single window a spectrum is broken into before hashing
    """
    def __init__(self, trials, len_trial, seed=5671, resolution=1, num_dalton=10):

        assert 0 < trials, f'trials variable needs to be greater then 1, was: {trials}'
        assert 0 < len_trial, f'length trial variable needs to be greater then 1, was: {trials}'

        # check
        if 0 < len_trial <= 32:
            self.V = tf.constant(
                np.expand_dims(np.array([np.power(2, i) for i in range(len_trial)]).astype(np.int32), 1))

        elif 32 < len_trial <= 64:
            warnings.warn(f"\nnum bits to hash set to: {len_trial}.\n" +
                          f"using int64 which might slow down computation significantly.")
            self.V = tf.constant(
                np.expand_dims(np.array([np.power(2, i) for i in range(len_trial)]).astype(np.int64), 1))

        else:
            raise ValueError(f"bit number per hash cannot be greater then 64 or smaller 1, was: {len_trial}.")

        self.trails = trials
        self.len_trial = len_trial
        self.seed = seed
        self.resolution = resolution
        self.num_dalton = num_dalton
        self.hash_ptr = pla.TimsHashGenerator(len_trial, trials, seed, resolution, num_dalton)
        self.hash_matrix = self.get_matrix()
        self.hash_tensor = self.tf_tensor()

    # fetch matrix from C++ implementation
    def get_matrix(self):
        return self.hash_ptr.getMatrixCopy()

    # create TF tensor (potentially GPU located) for hashing
    def tf_tensor(self):
        return tf.transpose(tf.constant(self.hash_matrix.astype(np.float32)))

    # create keys by random projection
    def calculate_keys(self, W: tf.Tensor):
        """

        :param W:
        """
        # generate signum
        S = (tf.sign(W @ self.hash_tensor) + 1) / 2

        if self.len_trial <= 32:
            # reshape into window, num_hashes, len_single_hash
            S = tf.cast(tf.reshape(S, shape=(S.shape[0], self.trails, self.len_trial)), tf.int32)

            # calculate int key from binary by base-transform
            H = tf.squeeze(S @ self.V)
            return H
        else:
            # reshape into window, num_hashes, len_single_hash
            S = tf.cast(tf.reshape(S, shape=(S.shape[0], self.trails, self.len_trial)), tf.int64)

            # calculate int key from binary by base-transform
            H = tf.squeeze(S @ self.V)
            return H

    def calculate_collisions(self, H: np.array, s: np.array, b: np.array):
        """

        :param H: HashMatrix
        :param s: vector of scans
        :param b: vector of bins
        :return: (bins, scans)
        """
        return self.hash_ptr.calculateCollisions(H, s, b)

    def filter_frame_auto_correlation(self, frame, min_intensity=1, overlapping=True, min_peaks=6):
        """

        :param min_peaks:
        :param frame:
        :param min_intensity:
        :param overlapping:
        :return:
        """
        s, b, W = frame.get_dense_windows(window_length=self.num_dalton, resolution=self.resolution,
                                          overlapping=overlapping, min_intensity=min_intensity, min_peaks=min_peaks)

        K = self.calculate_keys(W)
        mz_bins, scans = self.calculate_collisions(K, s, b)

        as_frame = frame.data()
        as_frame['bin'] = np.floor(as_frame['mz'] / self.num_dalton)
        as_frame['bin_overlapping'] = -np.floor((as_frame['mz'] + self.num_dalton / 2) / self.num_dalton)

        keep = pd.DataFrame({'mz_bin': mz_bins, 'scan': scans})

        non_overlapping = pd.merge(left=as_frame, right=keep, left_on=['scan', 'bin'], right_on=['scan', 'mz_bin'])
        overlapping = pd.merge(left=as_frame, right=keep, left_on=['bin_overlapping', 'scan'],
                               right_on=['mz_bin', 'scan'])

        f = pd.concat([non_overlapping,
                       overlapping]).drop_duplicates(['scan', 'mz'])[['frame', 'scan', 'mz', 'intensity']]

        filtered_frame = TimsFrame(None, f['frame'].values[0], f.scan.values, f.mz.values,
                                   f.intensity.values,
                                   np.zeros_like(f.mz.values).astype(np.int32),
                                   np.zeros_like(f.mz.values).astype(np.float32))

        return filtered_frame


class ReferencePattern:
    """
    """

    def __init__(self, mz_bin, mz_mono, mz_last_contrib, charge, spectrum, window_length=5, resolution=2):
        """
        :param mz_bin:
        :param mz_mono:
        :param mz_last_contrib:
        :param charge:
        :param spectrum:
        :param window_length:
        :param resolution:
        """
        self.mz_bin = mz_bin
        self.mz_mono = mz_mono
        self.charge = charge
        self.spec = spectrum
        self.window_length = window_length
        self.mz_last_contrib = mz_last_contrib
        self.patterns = self.get_rolling_patterns(resolution)

    def zero_indexed_sparse_vector(self, resolution, min_percent_contrib=1):
        """
        :param resolution:
        :param min_percent_contrib:
        :return:
        """
        assert 0 <= min_percent_contrib <= 100, f'percent contrib needs to be in [0, 100], was: \
        {min_percent_contrib}'

        binned_spectrum = self.spec.vectorize(resolution)
        indices, values = binned_spectrum.indices(), binned_spectrum.values()

        i = indices - np.min(indices)
        v = values / np.max(values)

        perc = min_percent_contrib / 100

        ret_i, ret_v = [], []
        for index, value in zip(i, v):
            if value >= perc:
                ret_i.append(index)
                ret_v.append(value)

        return np.array(ret_i), np.array(ret_v)

    def zero_indexed_dense_vector(self, resolution=2, min_percent_contrib=1):
        """
        :param resolution:
        :param min_percent_contrib:
        """
        indices, values = self.zero_indexed_sparse_vector(resolution, min_percent_contrib)
        zeros = np.zeros(int(self.window_length * np.power(10, resolution)) + 1)

        for i, v in zip(indices, values):
            if i < len(zeros):
                zeros[i] = v

        return zeros

    # TODO: remove rolled pattern reinsertion into front
    def get_rolling_patterns(self, resolution, min_percent_contrib=1):
        """
        :param resolution:
        :param min_percent_contrib:
        """
        p = self.zero_indexed_dense_vector(resolution, min_percent_contrib)

        r = []
        last = np.nonzero(p)[0][-1]

        for i in range(last):
            d = shift(p, i, cval=0)
            r.append(d)

        return np.array(r)

    def cosine_similarity(self, window, resolution=2, min_percent_contrib=1):
        """
        """
        dense_vectors = self.get_rolling_patterns(resolution, min_percent_contrib)
        c = cosine_similarity(dense_vectors, window.reshape(1, -1))
        best_sim = c[np.argmax(c)]

        return c, best_sim, dense_vectors[np.argmax(c)]

    def __repr__(self):
        return f'ReferencePattern(bin: {self.mz_bin}, mz mono: {self.mz_mono}, charge: \
    {self.charge}, last contrib: {self.mz_last_contrib})'


class IsotopeReferenceSearch:
    """
    """

    def __init__(self, reference_pattern, hasher):
        """
        """
        self.reference_pattern = reference_pattern
        self.hasher = hasher

        r_dict = {}

        for key in tqdm(self.reference_pattern, desc='calculating keys', ncols=100):
            mzs, charges, vectors = self.get_candidates_blocked(key)
            keys = hasher.calculate_keys(vectors)

            r_dict[key] = (mzs, charges, keys, vectors)

        self.key_dict = r_dict

    def get_candidates(self, b):
        """
        """
        if b in self.reference_pattern:
            return self.reference_pattern[b]

        else:
            return None

    def get_keys(self, b):
        """
        """
        if b in self.key_dict:
            return self.key_dict[b]

        else:
            return None

    def get_candidates_blocked(self, b, resolution=2):
        """
        """
        ref_pattern = self.get_candidates(b)
        c_list, m_list, bin_list = [], [], []

        for p in ref_pattern:

            M = p.get_rolling_patterns(resolution)

            b_inner = []

            for i in range(M.shape[0]):
                b_inner.append(p.mz_mono + i / np.power(10, resolution))

            C = np.repeat(p.charge, M.shape[0])
            c_list.append(C)
            m_list.append(M)
            bin_list.append(np.array(b_inner))

        C = np.concatenate(c_list)
        M = np.concatenate(m_list)
        B = np.concatenate(bin_list)

        return B, C, M

    def get_max_similarity(self, b, window, resolution=2):
        """
        """
        ref_pattern = self.get_candidates(b)
        B, C, M = self.get_candidates_blocked(b, resolution)

        SIM = cosine_similarity(M, window.reshape(1, -1))

        max_sim = np.max(SIM)
        argmax_sim = np.argmax(SIM)
        charge_state = C[argmax_sim]
        monoisotopic_mass = B[argmax_sim]

        return M[argmax_sim], max_sim, monoisotopic_mass, charge_state

    def find_isotope_patterns(self, frame: TimsFrame, min_intensity=100, min_peaks=5, overlapping=True,
                              window_length=4, min_cosim=0.6):
        """
        :param frame:
        :param min_intensity:
        :param min_peaks:
        :param overlapping:
        :param window_length:
        :param min_cosim:
        :return:
        """

        s, b, F = frame.get_dense_windows(
            window_length=self.hasher.num_dalton,
            resolution=self.hasher.resolution,
            min_intensity=min_intensity,
            min_peaks=min_peaks,
            overlapping=overlapping)

        WK = self.hasher.calculate_keys(F).numpy()
        F = F.numpy()

        r_list = []

        for scan, mz_bin, keys, vectors in zip(s, b, WK, F):
            r_list.append(self.calculate_window_collision(scan, mz_bin, keys, vectors, window_length=window_length,
                                                          min_cosim=min_cosim))

        A = np.hstack([np.expand_dims(s, axis=1), np.expand_dims(b, axis=1), np.array(r_list)])
        patterns = pd.DataFrame(A[A[:, 4] != -1], columns=['scan', 'bin', 'cosim', 'mz_mono', 'charge'])

        patterns['overlapping'] = patterns['bin'] < 0
        patterns['overlapping_inverted'] = patterns['bin'] > 0

        patterns['bin'] = np.abs(patterns['bin'])

        patterns['id'] = np.arange(patterns.shape[0])

        overlapping_false = patterns[patterns.overlapping == False]
        overlapping_true = patterns[patterns.overlapping]

        overlap_duplicates = pd.merge(left=overlapping_false, right=overlapping_true,
                                      left_on=['bin', 'overlapping', 'scan'],
                                      right_on=['bin', 'overlapping_inverted', 'scan'])

        overlap_duplicates['is_x_larger'] = overlap_duplicates.cosim_x > overlap_duplicates.cosim_y

        drop_ids = set(overlap_duplicates.apply(lambda r: r['id_y'] if r['is_x_larger'] else r['id_x'], axis=1).values)

        patterns = patterns[patterns.apply(lambda r: r['id'] not in drop_ids,
                                           axis=1)].drop(columns=['id', 'overlapping_inverted', 'overlapping'])

        return patterns

    def calculate_window_collision(self, scan, mz_bin, keys, dense_vector, window_length=4, min_cosim: float = 0.6):
        """

        :param scan:
        :param mz_bin:
        :param keys:
        :param dense_vector:
        :param window_length:
        :param min_cosim:
        :return:
        """

        is_overlapping = int(((np.sign(mz_bin) - 1) / - 2))
        mz_bin = np.abs(mz_bin)

        if mz_bin in self.key_dict:

            mz, charge, keys_ref, vectors = self.get_keys(np.abs(mz_bin))
            candidates = np.any(keys == keys_ref, axis=1)

            m_c, c_c, v_c = mz[candidates], charge[candidates], vectors[candidates]

            if len(m_c) > 0:

                real_cosim = cosine_similarity(v_c, dense_vector.reshape(1, -1))

                argmax_cosim = np.argmax(real_cosim)
                max_cosim = np.round(real_cosim[argmax_cosim][0], 2)
                max_m_c = m_c[argmax_cosim] - is_overlapping * (window_length / 2)
                max_c_c = c_c[argmax_cosim]

                if max_cosim > min_cosim:
                    return np.array([max_cosim, max_m_c, max_c_c])

                else:
                    return np.array([-1, -1, -1])

        return np.array([-1, -1, -1])
