import pytest
import numpy as np
from scipy.stats import norm
from collections import namedtuple

from proteolizarddata.data import MzSpectrum
from proteolizardalgo.isotopes import factorial, lam, weight, normal_pdf, iso, AveragineGenerator, MASS_NEUTRON

Peptide = namedtuple("Peptide", ["mass", "charge"])

@pytest.fixture(scope="module", params=[(969.487935,2),(1401.725205,3)])
def peptide(request):
    m, c = request.param
    return Peptide(m,c)

def test_factorial():
    assert factorial(0) == 1, "factorial edge case is invalid"
    assert factorial(1) == 1, "factorial edge case is invalid"
    assert factorial(5) == 120, "factorial function result is invalid"

def test_lam(peptide):
    # tested against status 16 feb 23
    assert np.isclose(lam(peptide.mass),0.000594*peptide.mass+(-0.03091)), "Calculated averagine lambda is unexpected"

def test_weight(peptide):
    # tested against status 16 feb 23
    mass = peptide.mass
    number_peaks = 5
    peaks = np.arange(number_peaks)
    iso_factorials = np.ones_like(peaks)
    for i, peak in enumerate(peaks):
        iso_factorials[i] = factorial(peak)
    averagine_lam = lam(peptide.mass)
    expected_weights = np.exp(-averagine_lam)*np.power(averagine_lam,peaks) / iso_factorials
    calculated_weights = weight(mass, peaks)
    assert np.allclose(expected_weights,calculated_weights), "Calculation of isotopic peak weights is invalid"

def test_normal_pdf(peptide):
    # tested against scipy
    mu = peptide.mass
    sigma = 0.01
    x = np.array([mu-sigma,mu,mu+sigma])
    calculated_ps = normal_pdf(x,mu,sigma)
    expected_ps = norm.pdf(x,loc=mu,scale=sigma)
    assert np.allclose(calculated_ps,expected_ps), "Calculation of normal probabilities is not corresponding to scipy implementation"

def test_iso(peptide):
    # tested against status 16 feb 23
    mass = peptide.mass
    charge = peptide.charge
    iso_distance = MASS_NEUTRON/charge
    sigma = 0.01
    amp = 1
    K = 7
    x = np.array([mass,mass+iso_distance,mass+2*iso_distance])
    calculated_intensities = iso(x,mass,charge,sigma,amp,K)

    # expected
    k = np.arange(K)
    means = ((mass + MASS_NEUTRON * k) / charge).reshape((1,-1))
    weights = weight(mass,k).reshape((1,-1))
    intensities = np.sum(weights*normal_pdf(x.reshape((-1,1)), means, sigma), axis= 1)
    expected_intensities =  intensities * amp

    np.allclose(calculated_intensities, expected_intensities)

class TestAveragineGenerator:

    @pytest.fixture(scope="class")
    def avg_gen(self):
        return AveragineGenerator()

    @pytest.mark.parametrize("centroided",[True,False])
    def test_generate_spectrum(self,avg_gen,peptide,centroided):
        mass = peptide.mass
        charge = peptide.charge
        s = avg_gen.generate_spectrum(mass,charge,-1,-1,centroided=centroided)
        # This test is unspecific. General fails shall be observed here
        assert isinstance(s,MzSpectrum)
