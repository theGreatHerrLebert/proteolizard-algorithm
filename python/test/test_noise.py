import pytest
import numpy as np

from proteolizardalgo.noise import detection_noise, baseline_shot_noise_window, baseline_shot_noise
from proteolizarddata.data import MzSpectrum

rng = np.random.default_rng(2023)

array_signal_1 = np.zeros(10)
array_signal_2 = rng.randint(0,1000,size=10)
mz_index = np.arange(10)*100

@pytest.fixture(scope="module", params=[array_signal_1,array_signal_2])
def signal_array(request):
    return request.param.copy()

@pytest.fixture(scope="module", params=[(array_signal_1,mz_index),(array_signal_2,mz_index)])
def signal_MzSpectrum(request):
    i = request.param[0]
    mz = request.param[1]
    return MzSpectrum(None,-1,-1,mz.copy(),i.copy())

@pytest.mark.parametrize("method",["poisson","normal"])
def test_detection_noise(signal_array, method):
    noised = detection_noise(signal_array, method).flatten()
    for idx in range(array_signal_1.flatten().size):
        if signal_array[idx] == 0:
            assert noised[idx] == 0, "Unexpected noise for Intensity=0 with default settings"

            @pytest.mark.parametrize("expected_noise",[0,10])
def test_baseline_shot_noise(signal_MzSpectrum, expected_noise):
    noised_spectrum = baseline_shot_noise(signal_MzSpectrum,window_size=1,expected_noise_peaks_per_Th=expected_noise,min_intensity=0)
    if expected_noise == 0:
        assert noised_spectrum.mz() == signal_MzSpectrum.mz(), "Input signal altered with zero noise"
        assert noised_spectrum.intensity() == signal_MzSpectrum.intensity(), "Input signal altered with zero noise"
    assert noised_spectrum.frame_id() == signal_MzSpectrum.frame_id(), "Spectrum's frame id was changed during noising"
    assert noised_spectrum.scan_id()== signal_MzSpectrum.scan_id(), "Spectrum's scan id was changed during noising"


