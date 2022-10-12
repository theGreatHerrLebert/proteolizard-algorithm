"""Tests for proteolizardalgo.feature_loader_dda module
"""
import pytest
import pandas as pd
from proteolizarddata.data import PyTimsDataHandleDDA
from proteolizardalgo.feature_loader_dda import FeatureLoaderDDA

@pytest.fixture(scope="module")
def data_path():
    # ! set data path to bruker experiment (.d folder) here
    return "/home/tim/Master/MassSpecDaten/M210115_001_Slot1-1_1_850.d"

@pytest.fixture(scope="module",params=[1000,2000])
def feature_id(request):
    return request.param

class TestFeatureLoaderDDA:

    @pytest.fixture(scope="class")
    def data_handle(self, data_path):
        return PyTimsDataHandleDDA(data_path)

    @pytest.fixture(scope="class")
    def feature_loader(self, data_handle,feature_id):
        return FeatureLoaderDDA(data_handle,feature_id)

    @pytest.mark.parametrize("ims_model",["gaussian","DBSCAN"])
    def test_load_hull_data_3d(self,feature_loader,ims_model):
        hull_data:pd.DataFrame = feature_loader.load_hull_data_3d(ims_model = ims_model)
        assert("Scan" in hull_data.columns, "Scan not in hull data")
        assert("Mz" in hull_data.columns,"Mz not in hull data")
        assert("Intensity" in hull_data.columns, "Intensity not in hull data")

    def test_get_precursor_summary(self, feature_loader):
        data:pd.DataFrame = feature_loader._get_precursor_summary()
        assert(data.shape==(1,4),"Precursor summary DataFrame has wrong shape")
        assert("MonoisotopicMz" in data.columns ,"No monoisotopic mz in precursor summary")
        assert("Charge" in data.columns ,"No charge in precursor summary")
        assert("ScanNumber" in data.columns ,"No scan number in precursor summary")
        assert("FrameID" in data.columns ,"No frame id in precursor summary")
