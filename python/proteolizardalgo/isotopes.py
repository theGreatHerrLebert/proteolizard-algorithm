from __future__ import annotations

import numpy as np
import math
from proteolizarddata.data import MzSpectrum

import numba

MASS_PROTON = 1.007276466583


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
def weight(mass: float, num_steps: int):
    """
    :param mass:
    :param num_steps:
    :return:
    """
    return np.exp(-lam(mass)) * np.power(lam(mass), num_steps) / factorial(num_steps)


@numba.jit(nopython=True)
def normal_pdf(x: float, mass: float, s: float = 0.001, inv_sqrt_2pi: float = 0.3989422804014327):
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
def iso(x: int, mass: float, charge: float, sigma: float, amp: float, K: int, mass_neutron: float = 1.008664916):
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
    acc = 0
    for k in range(0, K):
        mean = (mass + mass_neutron * k) / charge
        acc += weight(mass, k) * normal_pdf(x, mean, sigma)
    return amp * acc


@numba.jit(nopython=True)
def generate_pattern(lower_bound: float,
                     upper_bound: float,
                     step_size: float,
                     mass: float,
                     charge: float,
                     amp: float,
                     k: int,
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
    stop = upper_bound
    x = lower_bound

    mz_list, intensity_list = [], []

    while x < stop:
        intensity_list.append(iso(x, mass, charge, sigma, amp, k))
        mz_list.append(x)
        x = x + step_size

    return np.array(mz_list) + MASS_PROTON, np.array(intensity_list).astype(np.int32)
