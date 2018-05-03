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


def init_tensor(shape, name):
    ndims = len(shape)
    if ndims == 1:
        return T.matrix(name)
    elif ndims == 2:
        return T.tensor3(name)
    else:
        print('ndimerror@', name)
        return None


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
    def compile_dataset(filename='datatable_sm.csv', frac=1.):
        df = pd.read_csv(filename)
        df = df.loc[(df['duration']/60) < 600.]
        df = df.sample(frac=frac)
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
                data=df['mode'].values-1,
                dtype=DTYPE_FLOATX
            )
            f['mode'].attrs['dtype'] = VARIABLE_TYPE_CATEGORY

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
            f['purpose'].attrs['dtype'] = VARIABLE_TYPE_CATEGORY

            # ! avg_speed
            f.create_dataset(
                name='avg_speed/data',
                data=df['avg_speed'].values.reshape(len(df), 1)/18.76,
                dtype=DTYPE_FLOATX
            )
            f['avg_speed'].attrs['stdev'] = 18.76
            f['avg_speed'].attrs['mean'] = 23.077
            f['avg_speed'].attrs['dtype'] = VARIABLE_TYPE_INTEGER

            # ! duration
            f.create_dataset(
                name='duration/data',
                data=(df['duration'].values/60.).reshape(len(df), 1)/21.52,
                dtype=DTYPE_FLOATX
            )
            f['duration'].attrs['stdev'] = 21.52
            f['duration'].attrs['mean'] = 22.94
            f['duration'].attrs['dtype'] = VARIABLE_TYPE_INTEGER

            # ! n_coord
            f.create_dataset(
                name='n_coord/data',
                data=df['n_coord'].values.reshape(len(df), 1)/132.854,
                dtype=DTYPE_FLOATX
            )
            f['n_coord'].attrs['stdev'] = 132.854
            f['n_coord'].attrs['mean'] = 113.572
            f['n_coord'].attrs['dtype'] = VARIABLE_TYPE_INTEGER

            # ! trip_km
            f.create_dataset(
                name='trip_km/data',
                data=df['trip_km'].values.reshape(len(df), 1)/10.584,
                dtype=DTYPE_FLOATX
            )
            f['trip_km'].attrs['stdev'] = 10.584
            f['trip_km'].attrs['mean'] = 8.847
            f['trip_km'].attrs['dtype'] = VARIABLE_TYPE_INTEGER

            # ! interval
            stdev = [0.793, 0.557, 0.703, 0.576]
            f.create_dataset(
                name='interval/data',
                data=df[['startintervalsin', 'startintervalcos',
                         'endintervalsin', 'endintervalcos']].values,
                dtype=DTYPE_FLOATX
            )
            f['interval'].attrs['stdev'] = stdev
            f['interval'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! dow
            stdev = [0.718, 0.690]
            f.create_dataset(
                name='dow/data',
                data=df[['startdowsin', 'startdowcos']].values,
                dtype=DTYPE_FLOATX
            )
            f['dow'].attrs['stdev'] = stdev
            f['dow'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! dom
            stdev = [0.719, 0.679]
            f.create_dataset(
                name='dom/data',
                data=df[['startdomsin', 'startdomcos']].values,
                dtype=DTYPE_FLOATX
            )
            f['dom'].attrs['stdev'] = stdev
            f['dom'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! doy
            stdev = [0.113, 0.152]
            f.create_dataset(
                name='doy/data',
                data=df[['startdoysin', 'startdoycos']].values,
                dtype=DTYPE_FLOATX
            )
            f['doy'].attrs['stdev'] = stdev
            f['doy'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! startpoint
            stdev = [0.37042957, 0.59889923, 0.37042894, 0.59889845,
                     0.37407628, 0.60479492, 0.37553437, 0.60715244,
                     0.40837756, 0.66025098]
            f.create_dataset(
                name='startpoint/data',
                data=df[['pt0lat', 'pt0lon', 'pt1lat', 'pt1lon',
                         'pt2lat', 'pt2lon', 'pt3lat', 'pt3lon',
                         'pt4lat', 'pt4lon']].values,
                dtype=DTYPE_FLOATX
            )
            f['startpoint'].attrs['stdev'] = stdev
            f['startpoint'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! endpoint
            stdev = [0.37042987, 0.59890127, 0.37042994, 0.59890116,
                     0.37407723, 0.60479779, 0.37553557, 0.60715532,
                     0.40837907, 0.66025432]
            f.create_dataset(
                name='endpoint/data',
                data=df[['ptnm0lat', 'ptnm0lon', 'ptnm1lat', 'ptnm1lon',
                         'ptnm2lat', 'ptnm2lon', 'ptnm3lat', 'ptnm3lon',
                         'ptnm4lat', 'ptnm4lon']].values,
                dtype=DTYPE_FLOATX
            )
            f['endpoint'].attrs['stdev'] = stdev
            f['endpoint'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! startdistrict
            f.create_dataset(
                name='startdistrict/data',
                data=np.eye(df['startdistrictid'].values.max()+1)[
                     df['startdistrictid'].values].reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f.create_dataset(
                name='startdistrict/label',
                data=df['startdistrictid'].values,
                dtype=DTYPE_FLOATX
            )
            f['startdistrict'].attrs['dtype'] = VARIABLE_TYPE_CATEGORY

            # ! enddistrict
            f.create_dataset(
                name='enddistrict/data',
                data=np.eye(df['enddistrictid'].values.max()+1)[
                     df['enddistrictid'].values].reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f.create_dataset(
                name='enddistrict/label',
                data=df['enddistrictid'].values,
                dtype=DTYPE_FLOATX
            )
            f['enddistrict'].attrs['dtype'] = VARIABLE_TYPE_CATEGORY
