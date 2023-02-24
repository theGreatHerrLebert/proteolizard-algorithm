import numpy as np
import pytest

from proteolizardalgo.utility import normal_pdf
from scipy.stats import norm

@pytest.mark.parametrize("mu",[10,100,13.4])
def test_normal_pdf(mu):
    # tested against scipy
    sigma = 0.01
    x = np.array([mu-sigma,mu,mu+sigma])
    calculated_ps = normal_pdf(x,mu,sigma)
    expected_ps = norm.pdf(x,loc=mu,scale=sigma)
    assert np.allclose(calculated_ps,expected_ps), "Calculation of normal probabilities is not corresponding to scipy implementation"
