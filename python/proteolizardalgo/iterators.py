import numpy as np
from proteolizarddata.data import PyTimsDataHandleDDA
from proteolizardalgo.feature import FeatureBatch


class FeatureIterator:

    def __init__(self, feature_table, data_path, drop_first_n_percent=10, num_slices=50):
        assert 0 <= drop_first_n_percent <= 100, f"percent drop \
        must be between 0 and 1, was: {drop_first_n_percent}"

        self.name = data_path.split('/')[-1] if len(data_path.split('/')[-1]) > 1 else data_path.split('/')[-2]

        self.dh = PyTimsDataHandleDDA(data_path)

        self.feature_table = self.preprocess_table(feature_table, drop_first_n_percent)

        self.num_slices = num_slices

        self.__block_size = len(np.array_split(self.feature_table.rt_start.values, 40)[0])

        self.__block_counter = 0

        self.data_pair = self.get_feature_batch()

    def preprocess_table(self, feature_table, drop_first_n_percent):
        # remove first n percent of rows from table
        num_rt = feature_table.rt.values.shape[0]
        return feature_table.iloc[int((num_rt / 100) * drop_first_n_percent):]

    def get_feature_batch(self):
        next_n = self.feature_table.iloc[self.__block_size *
                                         self.__block_counter:self.__block_size * (self.__block_counter + 1)]

        rt_start = next_n.rt_start.values[0]
        rt_stop = next_n.rt_stop.values[-1]

        return FeatureBatch(next_n, self.dh.get_slice_rt_range(rt_start, rt_stop))

    def __iter__(self):
        return self.data_pair

    def __next__(self):
        data_pair = self.data_pair

        if self.__block_counter < self.num_slices - 1:
            self.__block_counter += 1
            self.data_pair = self.get_feature_batch()
            return data_pair

        raise StopIteration

    def __repr__(self):
        return f'FeatureIterator(dataset={self.name}, num_features={self.feature_table.shape[0]})'
