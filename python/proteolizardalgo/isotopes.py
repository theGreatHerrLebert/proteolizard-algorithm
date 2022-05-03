import numpy as np
import math

from proteolizarddata.data import MzSpectrum

# declare constants
SLOPE_LAM = 0.000594
INTERCEPT_LAM = -0.03091
MASS_NEUTRON = 1.008664916
INV_SQRT_2PI = 0.3989422804014327
SIGMA = 500 / (2.355 * 25000)


# linear dependency of mass
def lam(mass):
    """

    :param mass:
    :return:
    """
    return SLOPE_LAM * mass + INTERCEPT_LAM


#
def weight(mass, num_steps):
    """

    :param mass:
    :param num_steps:
    :return:
    """
    return math.exp(-lam(mass)) * math.pow(lam(mass), num_steps) / math.factorial(num_steps)


def normal_pdf(x, mass, s=0.001):
    """

    :param x:
    :param mass:
    :param s:
    :return:
    """
    a = (x - mass) / s
    return INV_SQRT_2PI / s * math.exp(-0.5 * a * a)


def iso(x, mass, charge, sigma, amp, K):
    """

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
        mean = (mass + MASS_NEUTRON * k) / charge
        acc += weight(mass, k) * normal_pdf(x, mean, sigma)
    return amp * acc


def generate_pattern(lower_bound, upper_bound, step_size, mass, charge, amp, k, sigma=SIGMA, resolution=2):
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
        x += step_size

    return MzSpectrum(None, -1, -1, mz_list, np.array(intensity_list).astype(np.int32)).to_resolution(resolution)

