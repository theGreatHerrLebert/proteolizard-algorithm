from __future__ import annotations
import warnings
import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod

from scipy.signal import argrelextrema
from proteolizarddata.data import MzSpectrum
from proteolizardalgo.utility import gaussian, exp_gaussian, normal_pdf
import numba

from proteolizardalgo.noise import detection_noise

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
def weight(mass: float, peak_nums: ArrayLike, normalize: bool = True):
    """
    :param mass:
    :param num_steps:
    :param normalize:
    :return:
    """
    factorials = np.zeros_like(peak_nums)
    norm = 1
    for i,k in enumerate(peak_nums):
        factorials[i] = factorial(k)
    weights = np.exp(-lam(mass)) * np.power(lam(mass), peak_nums) / factorials
    if normalize:
        norm = weights.sum()
    return weights/norm


@numba.jit(nopython=True)
def iso(x: ArrayLike, mass: float, charge: float, sigma: float, amp: float, K: int, step_size:float, add_detection_noise: bool = True, mass_neutron: float = MASS_NEUTRON):
    """
    :param mass_neutron:
    :param x:
    :param mass:
    :param charge:
    :param sigma:
    :param amp:
    :param K:4
    :param step_size:
    :param add_detection_noise:
    :return:
    """
    k = np.arange(K)
    means = ((mass + mass_neutron * k) / charge).reshape((1,-1))
    weights = weight(mass,k).reshape((1,-1))
    intensities = np.sum(weights*normal_pdf(x.reshape((-1,1)), means, sigma), axis= 1)*step_size
    if add_detection_noise:
        return detection_noise(intensities*amp)
    else:
        return intensities * amp


@numba.jit(nopython=True)
def numba_generate_pattern(lower_bound: float,
                     upper_bound: float,
                     mass: float,
                     charge: float,
                     amp: float,
                     k: int,
                     sigma: float = 0.008492569002123142,
                     resolution: int = 3):
    """
    :param lower_bound:
    :param upper_bound:
    :param mass:
    :param charge:
    :param amp:
    :param k:
    :param sigma:
    :param resolution:
    :return:
    """
    step_size = min(sigma/10,1/np.power(10,resolution))
    size = int((upper_bound-lower_bound)//step_size+1)
    mzs = np.linspace(lower_bound,upper_bound,size)
    intensities = iso(mzs,mass,charge,sigma,amp,k,step_size)

    return mzs + MASS_PROTON, intensities.astype(np.int64)

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
                         amp: float = 1e4, resolution: float = 3,
                         min_intensity: int = 5) -> (np.array, np.array):
        pass

    def generate_spectrum(self, mass: int, charge: int, frame_id: int, scan_id: int, k: int = 7,
                          amp :float = 1e4, resolution:float =3, min_intensity: int = 5, centroided: bool = True) -> MzSpectrum:

        if not 100 <= mass / charge <= 2000:
            warnings.warn(f"m/z should be between 100 and 2000, was: {mass / charge}")

        lb = mass / charge - .2
        ub = mass / charge + k + .2

        mz, i = numba_generate_pattern(lower_bound=lb, upper_bound=ub,
                                 mass=mass, charge=charge, amp=amp, k=k, resolution=resolution)

        if centroided:
            arg = argrelextrema(i, comparator=np.greater)[0]
            return MzSpectrum(None, frame_id, scan_id, mz[arg], i[arg]).to_resolution(resolution).filter(lb,ub,min_intensity)

        return MzSpectrum(None, frame_id, scan_id, mz, i).to_resolution(resolution).filter(lb,ub,min_intensity)
