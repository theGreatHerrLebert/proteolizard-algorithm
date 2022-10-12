import pandas as pd
from proteolizarddata.data import TimsSlice


class FeatureBatch:
    def __init__(self, feature_table: pd.DataFrame, raw_data: TimsSlice):
        """

        :param feature_table:
        :param raw_data:
        """
        self.feature_table = feature_table.sort_values(['rt_length'], ascending=False)
        self.raw_data = raw_data
        self.__feature_counter = 0
        self.current_row = self.feature_table.iloc[0]
        r = self.current_row
        self.current_feature = self.get_feature(r.mz - 0.1, r.mz + (r.num_peaks / r.charge) + 0.1,
                                                r.im_start,
                                                r.im_stop,
                                                r.rt_start,
                                                r.rt_stop)

    def __repr__(self):
        return f'FeatureBatch(features={self.feature_table.shape[0]}, slice={self.raw_data})'

    def get_feature(self, mz_min, mz_max, scan_min, scan_max, rt_min, rt_max, intensity_min=20):
        return self.raw_data.filter_ranged(mz_min=mz_min,
                                           mz_max=mz_max,
                                           scan_min=scan_min,
                                           scan_max=scan_max,
                                           intensity_min=intensity_min,
                                           rt_min=rt_min,
                                           rt_max=rt_max)

    def __iter__(self):
        return self.current_row, self.current_feature

    def __next__(self):
        feature = self.current_row
        data = self.current_feature

        if self.__feature_counter < self.feature_table.shape[0]:
            self.__feature_counter += 1
            self.current_row = self.feature_table.iloc[self.__feature_counter]
            r = self.current_row
            self.current_feature = self.get_feature(r.mz - 0.1,
                                                    r.mz + (r.num_peaks / r.charge) + 0.1,
                                                    r.im_start,
                                                    r.im_stop,
                                                    r.rt_start,
                                                    r.rt_stop)
            return feature, data

        raise StopIteration
