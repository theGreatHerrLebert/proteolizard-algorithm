import pandas as pd
from proteolizarddata.data import TimsSlice, PyTimsDataHandleDDA
from proteolizardalgo.iterators import FeatureIterator

from proteolizardvis.surface import TimsSurfaceVisualizer
from proteolizardvis.point import DDAPrecursorPointCloudVis


class SliceVessel:
    def __init__(self, data):
        self.filtered_data = data


if __name__ == '__main__':
    # name of the given .d file (without extension)
    ds = 'M210115_001_Slot1-1_1_850'
    # name of the maxquant extracted features, preprocessed
    # TODO: remove hard-coded path
    one_exp = pd.read_parquet(f'data/{ds}.parquet')

    # TODO: remove hard-coded path
    data = f'/media/hd01/CCSPred/{ds}.d/'
    fi = FeatureIterator(feature_table=one_exp, data_path=data, drop_first_n_percent=15)

    dh = PyTimsDataHandleDDA(data)
    batch = next(fi)

    while True:
        r, f = next(batch)
        sv = SliceVessel(f)
        surface_vis = TimsSurfaceVisualizer(data_loader=dh, data_filter=sv)
        surface_vis.display_widgets()
        point_cloud_vis = DDAPrecursorPointCloudVis(sv)
        point_cloud_vis.opacity_slider.value = .5
        point_cloud_vis.point_size_slider.value = 1.2
        point_cloud_vis.display_widgets()
        break
