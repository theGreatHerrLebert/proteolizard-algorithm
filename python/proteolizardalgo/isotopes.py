from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod

from scipy.signal import argrelextrema
from proteolizarddata.data import MzSpectrum
from proteolizardalgo.utility import gaussian, exp_gaussian
import numba

MASS_PROTON = 1.007276466583
MASS_NEUTRON = 1.008664916


@numba.jit(nopython=True)
def factorial(n: int):
    if n <= 0:
        return 1
    return n * factorial(n - 1)


# linear dependency of mass
@numba.jit(nopython=True)
def lam(mass: float, slope: float = 0.000594, intercept: float = -0.03091):
    """
    :param intercept:
    :param slope:
    :param mass:
    :return:
    """
    return slope * mass + intercept


@numba.jit(nopython=True)
def weight(mass: float, peak_nums: ArrayLike):
    """
    :param mass:
    :param num_steps:
    :return:
    """
    factorials = np.zeros_like(peak_nums)
    for i,k in enumerate(peak_nums):
        factorials[i] = factorial(k)
    return np.exp(-lam(mass)) * np.power(lam(mass), peak_nums) / factorials


@numba.jit(nopython=True)
def normal_pdf(x: ArrayLike, mass: float, s: float = 0.001, inv_sqrt_2pi: float = 0.3989422804014327):
    """
    :param inv_sqrt_2pi:
    :param x:
    :param mass:
    :param s:
    :return:
    """
    a = (x - mass) / s
    return inv_sqrt_2pi / s * np.exp(-0.5 * a * a)


@numba.jit(nopython=True)
def iso(x: ArrayLike, mass: float, charge: float, sigma: float, amp: float, K: int, mass_neutron: float = MASS_NEUTRON):
    """
    :param mass_neutron:
    :param x:
    :param mass:
    :param charge:
    :param sigma:
    :param amp:
    :param K:
    :return:
    """

    k = np.arange(K)
    means = ((mass + mass_neutron * k) / charge).reshape((1,-1))
    weights = weight(mass,k).reshape((1,-1))
    intensities = np.sum(weights*normal_pdf(x.reshape((-1,1)), means, sigma), axis= 1)
    return intensities * amp


@numba.jit(nopython=True)
def numba_generate_pattern(lower_bound: float,
                     upper_bound: float,
                     step_size: float,
                     mass: float,
                     charge: float,
                     amp: float,
                     k: int,
                     min_intensity: int,
                     sigma: float = 0.008492569002123142,
                     resolution: int = 2):
    """
    :param lower_bound:
    :param upper_bound:
    :param step_size:
    :param mass:
    :param charge:
    :param amp:
    :param k:
    :param sigma:
    :param resolution:
    :return:
    """
    size = int((upper_bound-lower_bound)//step_size+1)
    mzs = np.linspace(lower_bound,upper_bound,size)
    intensities = iso(mzs,mass,charge,sigma,amp,k)
    mz_filtered = mzs[intensities >= min_intensity]
    i_filtered = intensities[intensities >= min_intensity]

    return mz_filtered + MASS_PROTON, i_filtered.astype(np.int64)

@numba.jit(nopython=True)
def centroid_pattern(mzs:ArrayLike,intensities:ArrayLike, comp_range:int = 1):
    # this function is adapted from `scipy.signal.argrelextrema`
    # reimplemented for numba speed up


    n = intensities.size

    # numba does nor support "clip" mode in `np.take`
    # neither `np.pad`, so create a padded array by hand
    start_intensity = intensities[0]
    end_intensity = intensities[n-1]
    pad_left = np.repeat(start_intensity,comp_range)
    pad_right = np.repeat(end_intensity,comp_range)
    padded_intensities = np.concatenate((pad_left,intensities,pad_right))

    is_local_max = np.ones(n,dtype=np.bool8)
    idx = np.arange(comp_range, n+comp_range)
    middle = padded_intensities.take(idx)
    for i in range(comp_range):
        left = padded_intensities.take(idx-1)
        right = padded_intensities.take(idx+1)
        is_local_max &= (middle > left)
        is_local_max &= (middle > right)
        if not np.any(is_local_max):
            break
    arg_local_max = np.nonzero(is_local_max)[0]

    return mzs[arg_local_max], intensities[arg_local_max]

@numba.jit
def create_initial_feature_distribution(num_rt: int, num_im: int,
                                        rt_lower: float = -9,
                                        rt_upper: float = 18,
                                        im_lower: float = -4,
                                        im_upper: float = 4,
                                        distr_im=gaussian,
                                        distr_rt=exp_gaussian) -> np.array:

    I = np.ones((num_rt, num_im)).astype(np.float32)

    for i, x in enumerate(np.linspace(im_lower, im_upper, num_im)):
        for j, y in enumerate(np.linspace(rt_lower, rt_upper, num_rt)):
            I[j, i] *= (distr_im(x) * distr_rt(y))

    return I


class IsotopePatternGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_pattern(self, mass: float, charge: int) -> (np.array, np.array):
        pass

    @abstractmethod
    def generate_spectrum(self, mass: int, charge: int, min_intensity: int) -> MzSpectrum:
        pass


class AveragineGenerator(IsotopePatternGenerator):
    def __init__(self):
        super(AveragineGenerator).__init__()

    def generate_pattern(self, mass: float, charge: int, k: int = 7,
                         amp: float = 1e4, step_size: float = 1e-3,
                         min_intensity: int = 5) -> (np.array, np.array):
        assert 100 <= mass / charge <= 2000, f"m/z should be between 100 and 2000, was: {mass / charge}"

        lb = mass / charge - .2
        ub = mass / charge + k + .2

        mz, i = numba_generate_pattern(lower_bound=lb, upper_bound=ub, step_size=step_size,
                                 mass=mass, charge=charge, amp=amp, k=k, min_intensity=min_intensity)

        return mz, i

    def generate_spectrum(self, mass: int, charge: int, frame_id: int, scan_id: int, k: int = 7,
                          min_intensity: int = 5, centroided: bool = True) -> MzSpectrum:

        mz, i = self.generate_pattern(mass, charge, min_intensity=min_intensity, k=k)

        if centroided:
            arg = argrelextrema(i, comparator=np.greater)[0]
            return MzSpectrum(None, frame_id, scan_id, mz[arg], i[arg])

        return MzSpectrum(None, frame_id, scan_id, mz, i)
