"""Tests for proteolizardalgo.feature_loader_dda module
"""
import pytest
from proteolizarddata.data import PyTimsDataHandleDDA
from proteolizardalgo.feature_loader_dda import FeatureLoaderDDA

@pytest.fixture(scope="module")
def data_path():
    # ! set data path to bruker experiment (.d folder) here
    return "/home/tim/Master/MassSpecDaten/M210115_001_Slot1-1_1_850.d"

@pytest.fixture(scope="module",params=[1000,2000])
def feature_id(id):
    return id

class TestFeatureLoaderDDA:

    @pytest.fixture(scope="class")
    def data_handle(self, data_path):
        return PyTimsDataHandleDDA(data_path)

    @pytest.fixture(scope="class")
    def feature_loader(self, data_handle,feature_id):
        return FeatureLoaderDDA(data_handle,feature_id)

    def test_load_hull_data_3d(self):
        pass

    def test_fetch_precursor(self):
        pass

    def test_get_precursor_summary(self):
        pass

    def test_get_scan_boundaries(self):
        pass

    def test_get_num_peaks(self):
        pass

    def test_get_monoisotopic_profile(self):
        pass
