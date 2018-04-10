import h5py
import theano
import numpy as np
import pandas as pd
import theano.tensor as T

VARIABLE_TYPE_BINARY = 'binary'
VARIABLE_TYPE_REAL = 'real'
VARIABLE_TYPE_CATEGORY = 'category'
VARIABLE_TYPE_INTEGER = 'integer'
DTYPE_FLOATX = theano.config.floatX


class Metric(object):
    def __init__(self):
        pass

    def __confusion_matrix(self, y1, y2, num_cat):
        conf_mat = np.zeros((num_cat, num_cat), DTYPE_FLOATX)
        for i, j in zip(y1, y2):
            conf_mat[i][j] += 1
        return conf_mat

    def __histogram(self, labels, num_cat):
        hist = np.zeros((num_cat))
        for r in labels:
            hist[r] += 1
        return hist

    @staticmethod
    def kappa(y1, y2, num_cat=None, weighted='quadratic'):
        if num_cat is None:
            min_r = min(min(y1), min(y2))
            max_r = max(max(y1), max(y2))
            num_cat = max_r - min_r + 1

        conf_mat = self.__confusion_matrix(y1, y2)
        num_labels = len(conf_mat)
        num_items = float(len(y1))
        hist1 = self.__histogram(y1, num_cat)
        hist2 = self.__histogram(y2, num_cat)
        numerator = 0.
        denominator = 0.

        for i in range(num_labels):
            for j in range(num_labels):
                expected_count = hist1[i] * hist2[j] / num_items
                d = np.power(i-j, 2.) / np.power(num_labels, 2.)
                numerator += d * conf_mat[i][j] / num_items
                denominator += d * expected_count / num_items

        return 1. - numerator / denominator


class SetupH5PY(object):
    def __init__(self):
        pass

    @staticmethod
    def load_dataset(filename='data.h5'):
        f = h5py.File(filename, 'r')
        return f

    @staticmethod
    def compile_dataset(filename='datatable_sm.csv'):
        df = pd.read_csv(filename)
        with h5py.File('data.h5', 'w') as f:
            f.attrs['n_rows'] = len(df)

            # ! mode
            f.create_dataset(
                name='mode/data',
                data=np.eye(df['mode'].values.max())[
                     df['mode'].values-1].reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f.create_dataset(
                name='mode/label',
                data=df.mode.values-1,
                dtype=DTYPE_FLOATX
            )
            f['mode'].attrs['dtype'] = 'category'

            # ! purpose
            f.create_dataset(
                name='purpose/data',
                data=np.eye(df['purpose'].values.max())[
                     df['purpose'].values-1].reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f.create_dataset(
                name='purpose/label',
                data=df['purpose'].values-1,
                dtype=DTYPE_FLOATX
            )
            f['purpose'].attrs['dtype'] = 'category'

            # ! avg_speed
            f.create_dataset(
                name='avg_speed/data',
                data=df['avg_speed']values.reshape(len(df), 1, -1)/18.726,
                dtype=DTYPE_FLOATX
            )
            f['avg_speed'].attrs['stdev'] = 18.726
            f['avg_speed'].attrs['mean'] = 23.277
            f['avg_speed'].attrs['dtype'] = 'real'

            # ! duration
            f.create_dataset(
                name='duration/data',
                data=(df['duration']values/60).reshape(len(df), 1, -1)/131.666,
                dtype=DTYPE_FLOATX
            )
            f['duration'].attrs['stdev'] = 131.666
            f['duration'].attrs['mean'] = 24.34
            f['duration'].attrs['dtype'] = 'real'

            # ! n_coord
            f.create_dataset(
                name='n_coord/data',
                data=df['n_coord']values.reshape(len(df), 1, -1)/132.854,
                dtype=DTYPE_FLOATX
            )
            f['n_coord'].attrs['stdev'] = 132.854
            f['n_coord'].attrs['mean'] = 113.572
            f['n_coord'].attrs['dtype'] = 'real'

            # ! trip_km
            f.create_dataset(
                name='trip_km/data',
                data=df['trip_km']values.reshape(len(df), 1, -1)/10.584,
                dtype=DTYPE_FLOATX
            )
            f['trip_km'].attrs['stdev'] = 10.584
            f['trip_km'].attrs['mean'] = 8.847
            f['trip_km'].attrs['dtype'] = 'real'

            # ! startinterval_sin
            f.create_dataset(
                name='startintervalsin/data',
                data=df['startintervalsin']values.reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f['startintervalsin'].attrs['stdev'] = 0.725
            f['startintervalsin'].attrs['mean'] = -0.273
            f['startintervalsin'].attrs['dtype'] = 'real'

            # ! startinterval_cos
            f.create_dataset(
                name='startintervalcos/data',
                data=df['startintervalcos']values.reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f['startintervalcos'].attrs['stdev'] = 0.558
            f['startintervalcos'].attrs['mean'] = -0.296
            f['startintervalcos'].attrs['dtype'] = 'real'

            # ! endinterval_sin
            f.create_dataset(
                name='endintervalsin/data',
                data=df['endintervalsin']values.reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f['endintervalsin'].attrs['stdev'] = 0.705
            f['endintervalsin'].attrs['mean'] = -0.304
            f['endintervalsin'].attrs['dtype'] = 'real'

            # ! endinterval_cos
            f.create_dataset(
                name='endintervalcos/data',
                data=df['endintervalcos']values.reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f['endintervalcos'].attrs['stdev'] = 0.580
            f['endintervalcos'].attrs['mean'] = -0.270
            f['endintervalcos'].attrs['dtype'] = 'real'

            # ! dow
            f.create_dataset(
                name='dowsin/data',
                data=df['startdowsin']values.reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f['dowsin'].attrs['stdev'] = 0.668
            f['dowsin'].attrs['mean'] = -0.149
            f['dowsin'].attrs['dtype'] = 'real'

            f.create_dataset(
                name='dowcos/data',
                data=df['startdowcos']values.reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f['dowcos'].attrs['stdev'] = 0.725
            f['dowcos'].attrs['mean'] = -0.273
            f['dowcos'].attrs['dtype'] = 'real'

            # ! dom
            f.create_dataset(
                name='domsin/data',
                data=df['startdomsin']values.reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f['domsin'].attrs['stdev'] = 0.654
            f['domsin'].attrs['mean'] = -0.394
            f['domsin'].attrs['dtype'] = 'real'

            # ! dom
            f.create_dataset(
                name='domcos/data',
                data=df['startdomcos']values.reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f['domcos'].attrs['stdev'] = 0.645
            f['domcos'].attrs['mean'] = 0.021
            f['domcos'].attrs['dtype'] = 'real
