import pandas as pd
import unittest
import numpy as np


class TestData (unittest.TestCase):
    def test_data (self):
        url = '../data/preprocessed/data.parquet.gzip'
        df = pd.read_parquet(url)

        for column in df.columns:
            self.assertEqual(df[column].isna().sum(),
                             np.int64(0),
                             f'MSNO: {column}')


    def test_columns (self):
        url = '../data/preprocessed/data.parquet.gzip'
        df = pd.read_parquet(url)

        self.assertEqual(len(df.columns),
                         15,
                         'WNC detected')


if __name__ == '__main__':
    unittest.main()
