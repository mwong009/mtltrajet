import h5py
import os
import time
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
    if ndims == 0:
        return T.ivector(name)
    if ndims == 1:
        return T.matrix(name)
    elif ndims == 2:
        return T.tensor3(name)
    else:
        print('ndimerror@', name)
        return None


def get_time(t0):
    minutes, seconds = divmod(time.time() - t0, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds


def save_samples(path, name, x, y, samples, steps, i):
    filepath = '{0:s}{1:s}_{2:d}draws_epoch{3:d}.csv'.format(
        path, name, steps, i)
    df = pd.DataFrame()
    if not os.path.isdir(path):
        os.mkdir(path)
    for target in (y + x):
        df[target] = ''

    for j, (sample, target) in enumerate(zip(samples, (x + y))):
        target_name = target.name.strip('/')
        if target.attrs['dtype'] == VARIABLE_DTYPE_CATEGORY:
            # classification
            gen_class = (np.argmax(sample, axis=-1) + 1).squeeze()
            df[target_name] = gen_class
        else:
            if sample.shape[-1] > 1:
                df[target_name] = np.round(sample, 3).tolist()
            else:
                df[target_name] = np.round(sample, 3)

    with open(filepath, 'w+') as f:
        df.to_csv(f, header=True, index=False)


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
    def loglikelihood(prob, label):
        """
        get_sample_loglikelihood func
            Approximation to the reconstruction error

        Parameters
        ----------
        prob : `[T.shared]`
            list of precomputed "logits"
        label : `[T.shared]`
            list of output "labels"

        Returns
        -------
        nll : `scalar`
            value of the negative log likelihood
        """
        nll = -T.sum((T.log(prob)[T.arange(label.shape[0]), label]))
        return nll

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

    # @staticmethod
    # def compile_dataset(filename='datatable.csv', frac=1., dropout=1.,
    #                     h5file='data_valid.h5'):
    def compile_dataset(filename='datatable_sm.csv', frac=1., dropout=1.):
        df = pd.read_csv(filename)
        df = df.loc[(df['duration']/60) < 600.]
        if filename == 'datatable.csv':
            df = df.loc[df['mode'] == 0]
        # df = df.sample(frac=frac)
        if dropout < 1:
            for column in df.columns:
                df[column] = df[column] * np.random.binomial(
                    1, dropout, len(df[column]))
        # with h5py.File(h5file, 'w') as f:
        with h5py.File('data.h5', 'w') as f:
            f.attrs['n_rows'] = len(df)

            # ! mode (max=6)
            one_hot = np.eye(7)
            f.create_dataset(
                name='mode/data',
                data=one_hot[df['mode']][..., 1:].reshape([len(df), 1, -1]),
                dtype=DTYPE_FLOATX
            )
            f.create_dataset(
                name='mode/label',
                data=df['mode'].values-1,
                dtype=DTYPE_FLOATX
            )
            f['mode'].attrs['dtype'] = VARIABLE_TYPE_CATEGORY

            # ! purpose (max=9)
            one_hot = np.eye(10)
            f.create_dataset(
                name='purpose/data',
                data=one_hot[df['purpose']][..., 1:].reshape([len(df), 1, -1]),
                dtype=DTYPE_FLOATX
            )
            f.create_dataset(
                name='purpose/label',
                data=df['purpose'].values-1,
                dtype=DTYPE_FLOATX
            )
            f['purpose'].attrs['dtype'] = VARIABLE_TYPE_CATEGORY

            # ! startdistrict (max=34)
            one_hot = np.eye(35)
            f.create_dataset(
                name='startdistrict/data',
                data=one_hot[df['startdistrictid']].reshape([len(df), 1, -1]),
                dtype=DTYPE_FLOATX
            )
            f.create_dataset(
                name='startdistrict/label',
                data=df['startdistrictid'].values,
                dtype=DTYPE_FLOATX
            )
            f['startdistrict'].attrs['dtype'] = VARIABLE_TYPE_CATEGORY

            # ! enddistrict (max=34)
            one_hot = np.eye(35)
            f.create_dataset(
                name='enddistrict/data',
                data=one_hot[df['enddistrictid']].reshape([len(df), 1, -1]),
                dtype=DTYPE_FLOATX
            )
            f.create_dataset(
                name='enddistrict/label',
                data=df['enddistrictid'].values,
                dtype=DTYPE_FLOATX
            )
            f['enddistrict'].attrs['dtype'] = VARIABLE_TYPE_CATEGORY

            # ! avg_speed
            f.create_dataset(
                name='avg_speed/data',
                data=(df['avg_speed'].values/18.76).reshape([len(df), 1]),
                dtype=DTYPE_FLOATX
            )
            f['avg_speed'].attrs['stdev'] = 18.76
            f['avg_speed'].attrs['mean'] = 23.077
            f['avg_speed'].attrs['dtype'] = VARIABLE_TYPE_INTEGER

            # ! duration
            f.create_dataset(
                name='duration/data',
                data=(df['duration'].values/60./21.52).reshape([len(df), 1]),
                dtype=DTYPE_FLOATX
            )
            f['duration'].attrs['stdev'] = 21.52
            f['duration'].attrs['mean'] = 22.94
            f['duration'].attrs['dtype'] = VARIABLE_TYPE_INTEGER

            # ! n_coord
            f.create_dataset(
                name='n_coord/data',
                data=(df['n_coord'].values/132.854).reshape([len(df), 1]),
                dtype=DTYPE_FLOATX
            )
            f['n_coord'].attrs['stdev'] = 132.854
            f['n_coord'].attrs['mean'] = 113.572
            f['n_coord'].attrs['dtype'] = VARIABLE_TYPE_INTEGER

            # ! trip_km
            f.create_dataset(
                name='trip_km/data',
                data=(df['trip_km'].values/10.584).reshape([len(df), 1]),
                dtype=DTYPE_FLOATX
            )
            f['trip_km'].attrs['stdev'] = 10.584
            f['trip_km'].attrs['mean'] = 8.847
            f['trip_km'].attrs['dtype'] = VARIABLE_TYPE_INTEGER

            # ! interval
            f.create_dataset(
                name='interval/data',
                data=df[['startintervalsin', 'startintervalcos',
                         'endintervalsin', 'endintervalcos']].values,
                dtype=DTYPE_FLOATX
            )
            f['interval'].attrs['stdev'] = [0.793, 0.557, 0.703, 0.576]
            f['interval'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! dow
            f.create_dataset(
                name='dow/data',
                data=df[['startdowsin', 'startdowcos']].values,
                dtype=DTYPE_FLOATX
            )
            f['dow'].attrs['stdev'] = [0.718, 0.690]
            f['dow'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! dom
            f.create_dataset(
                name='dom/data',
                data=df[['startdomsin', 'startdomcos']].values,
                dtype=DTYPE_FLOATX
            )
            f['dom'].attrs['stdev'] = [0.719, 0.679]
            f['dom'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! doy
            f.create_dataset(
                name='doy/data',
                data=df[['startdoysin', 'startdoycos']].values,
                dtype=DTYPE_FLOATX
            )
            f['doy'].attrs['stdev'] = [0.113, 0.152]
            f['doy'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! startpoint
            f.create_dataset(
                name='startpoint/data',
                data=df[['pt0lat', 'pt0lon', 'pt1lat', 'pt1lon',
                         'pt2lat', 'pt2lon', 'pt3lat', 'pt3lon',
                         'pt4lat', 'pt4lon']].values,
                dtype=DTYPE_FLOATX
            )
            f['startpoint'].attrs['stdev'] = [0.3704, 0.5988, 0.3704, 0.5988,
                                              0.3740, 0.6047, 0.3755, 0.6071,
                                              0.4083, 0.6602]
            f['startpoint'].attrs['dtype'] = VARIABLE_TYPE_REAL

            # ! endpoint
            f.create_dataset(
                name='endpoint/data',
                data=df[['ptnm0lat', 'ptnm0lon', 'ptnm1lat', 'ptnm1lon',
                         'ptnm2lat', 'ptnm2lon', 'ptnm3lat', 'ptnm3lon',
                         'ptnm4lat', 'ptnm4lon']].values,
                dtype=DTYPE_FLOATX
            )
            f['endpoint'].attrs['stdev'] = [0.3704, 0.5989, 0.3704, 0.5989,
                                            0.3740, 0.6047, 0.3755, 0.6071,
                                            0.4083, 0.6602]
            f['endpoint'].attrs['dtype'] = VARIABLE_TYPE_REAL
