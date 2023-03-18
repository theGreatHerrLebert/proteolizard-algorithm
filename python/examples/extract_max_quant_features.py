import os
import sys

import pandas as pd

from proteolizardalgo.utils.helpers import preprocess_max_quant_evidence, preprocess_max_quant_sequence
from tqdm import tqdm

if __name__ == '__main__':
    if not 1 < len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <path/to/mq/features>')
        exit(1)

    path = sys.argv[1]

    for file in tqdm(os.listdir(path)):
        if file.find('.txt') != -1:
            data = pd.read_table(path + file, low_memory=False)
            processed_data = preprocess_max_quant_evidence(data)
            processed_data['sequence_tokenized'] = processed_data.apply(
                lambda r: preprocess_max_quant_sequence(r['sequence']), axis=1)
            single_experiments = list(set(processed_data['raw']))

            for exp in single_experiments:
                single_exp = processed_data[processed_data.raw == exp]
                single_exp = single_exp.sort_values(['rt', 'intensity'])
                single_exp.to_parquet(f'data/{exp}.parquet', index=False)
