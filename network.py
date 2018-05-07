import theano
import pickle
import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import matplotlib.pyplot as plt
from collections import OrderedDict
from optimizers import Optimizers
from pylab import rcParams

DTYPE_FLOATX = theano.config.floatX


class Network(object):
    def __init__(self, name, hyper, load_params=False):

        if load_params:
            try:
                with open(name + '.params', 'rb') as f:
                    model_values, hyper, curves = pickle.load(f)
            except IOError as e:
                print("Error opening file: ", e)
        else:
            model_values = {}
            curves = {
                'CD error': [],
                'MSE 1': [],
                'MSE 2': [],
                'log likelihood': [],
                'validation error': []
            }
            std_err = {}

        # initialize random number generator
        self.np_rng = np.random.RandomState(hyper['seed'])
        self.theano_rng = RandomStreams(hyper['seed'])

        self.name = name
        self.model_values = model_values
        self.hyperparameters = hyper
        self.monitoring_curves = curves
        self.params = OrderedDict()
        self.params_shp = OrderedDict()

        # Optimizer
        opt = Optimizers()
        if hyper['learner'] == 'amsgrad':
            self.update_opt = opt.adam_updates
        elif hyper['learner'] == 'momentum':
            self.update_opt = opt.momentum_updates
        elif hyper['learner'] == 'rmsprop':
            self.update_opt = opt.rmsprop_updates
        else:
            self.update_opt = opt.sgd_updates

    def save_params(self, epoch):
        """
        save_params func
            Saves model parameter values to a pickle file. To read
            params, unpickle and reshape.

        """
        path = 'params/{0:s}_epoch{1:d}.params'.format(self.name, epoch)
        hyper = self.hyperparameters
        curves = self.monitoring_curves
        model_values = {}
        # evaluating tensor shared variable to numpy array
        for param_name, param in self.params.items():
            model_values[param_name] = param.eval().reshape(
                self.params_shp[param_name])

        to_file = model_values, hyper, curves
        with open(path, 'wb') as f:
            pickle.dump(to_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_curves(self, curves=None):

        if curves is None:
            curves = self.monitoring_curves
        rcParams['axes.xmargin'] = 0
        rcParams['axes.ymargin'] = 0
        rcParams['figure.figsize'] = (12.8, 9.6)
        fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(nrows=2, ncols=2)
        ax1.set(title='CD loss', xlabel='iterations')
        ax1.plot(
            *zip(*curves['CD error']),
            linewidth=0.5,
            color='C0'
        )
        ax2.set(title='log likelihood loss', xlabel='iterations')
        ax2.plot(
            *zip(*curves['log likelihood']),
            linewidth=0.5,
            color='C1'
        )
        ax3.set(title='MSE 1 loss', xlabel='iterations')
        ax3.plot(
            *zip(*curves['MSE 1']),
            linewidth=0.5,
            color='C2'
        )
        ax4.set(title='MSE 2 loss', xlabel='iterations')
        ax4.plot(
            *zip(*curves['MSE 2']),
            linewidth=0.5,
            color='C3'
        )
        fig.set
        fig.savefig(self.name)
        plt.close()
