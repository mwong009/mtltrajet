import theano
import time
import pickle
import numpy as np
import pandas as pd
import theano.tensor as T
from theano import shared
from collections import OrderedDict
from optimizers import Optimizers
from utility import SetupH5PY, init_tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# CONSTANTS
VARIABLE_DTYPE_BINARY = 'binary'
VARIABLE_DTYPE_REAL = 'real'
VARIABLE_DTYPE_CATEGORY = 'category'
VARIABLE_DTYPE_INTEGER = 'integer'
DTYPE_FLOATX = theano.config.floatX


def main(data):
    # optimizer
    opt = Optimizers()

    # sampler
    theano_rng = RandomStreams(999)

    # import dataset
    n_samples = data.attrs['n_rows']
    lr = 1e-3
    batch_size = 128

    x_data = [
        data['purpose'], data['avg_speed'],
        data['duration'], data['trip_km'],
        data['n_coord'], data['interval'],
        data['dow'],
        data['startdistrict'],
        data['enddistrict']
    ]

    y_data = [data['mode']]

    params = OrderedDict()
    params_shp = OrderedDict()

    output = []
    input = []
    asc_params = []
    asc_params_m = []
    beta_params_f = []
    beta_params_s = []
    beta_params_sf = []
    beta_params = []
    beta_params_m = []

    for var in y_data:
        name = 'asc_' + var.name.strip('/')
        asc_shp = var['data'][:].squeeze().shape[1:]
        print('y', name, asc_shp)

        output.append(init_tensor((), name))

        mask = np.ones(asc_shp, DTYPE_FLOATX)
        mask[-1] = 0.
        asc_value = np.zeros(asc_shp, DTYPE_FLOATX) * mask

        asc_params.append(shared(asc_value, name))
        asc_params_m.append(shared(mask, name + '_mask'))

        params[name] = asc_params[-1]
        params_shp[name] = asc_shp

    for var in x_data:
        name = 'beta_' + var.name.strip('/')
        shp = var['data'].shape[1:] + asc_shp
        print('x', name, shp)

        input.append(init_tensor(var['data'].shape[1:], name))

        mask = np.ones(shp, DTYPE_FLOATX)
        mask[..., -1] = 0.
        mask = mask.flatten()
        beta_value = np.zeros(np.prod(shp), DTYPE_FLOATX) * mask
        sigma_value = 1e-3 * np.ones(np.prod(shp), DTYPE_FLOATX) * mask

        beta_params_f.append(shared(beta_value, name))
        beta_params_sf.append(shared(sigma_value, name + '_sigma'))

        beta_params.append(T.reshape(beta_params_f[-1], shp))
        beta_params_s.append(T.reshape(beta_params_sf[-1], shp))
        beta_params_m.append(shared(mask, name + '_mask'))

        params[name] = beta_params_f[-1]
        params[name + '_sigma'] = beta_params_sf[-1]
        params_shp[name] = shp
        params_shp[name + '_sigma'] = shp

    # compute the utility function
    utility = 0.
    for x, b, s in zip(input, beta_params, beta_params_s):

        normal_sample = b[..., None] + T.sqr(s)[..., None] * theano_rng.normal(
            size=b.eval().shape + (1,),
            avg=0.,
            std=1.,
            dtype=DTYPE_FLOATX
        )

        # normal_sample = b[..., None] + T.sqr(s)[..., None] * 0.

        ax = [np.arange(x.ndim)[1:], np.arange(b.ndim)[:-1]]
        utility += T.tensordot(x, normal_sample, axes=ax)

    for y, asc in zip(output, asc_params):
        utility += asc[None, ..., None]
        (d1, d2, d3) = utility.shape
        utility = utility.reshape((d1 * d3, d2))
        p_y_given_x = T.nnet.softmax(utility)
        # cost = -T.sum(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
        nll = T.log(p_y_given_x).reshape((d3, d1, d2))
        nll = nll[:, T.arange(y.shape[0]), y]
        cost = -T.sum(T.mean(nll, axis=0))

    gparams = asc_params + beta_params_f + beta_params_sf
    grads = T.grad(cost, gparams)

    # mask gradient updates
    mask = asc_params_m + beta_params_m + beta_params_m
    for j, g in enumerate(grads):
        grads[j] = g * mask[j]

    # create list of updates to theano function
    updates = opt.sgd_updates(gparams, grads, lr)

    # stderrs = []
    # hessian = T.hessian(cost=cost, wrt=gparams)
    # stderr = [T.sqrt(f) for f in [T.diag(1. / h) for h in hessian]]
    # stderrs.extend(stderr)

    tensors = input + output
    shared_x = [shared(var['data'][:], borrow=True) for var in x_data]
    shared_y = [T.cast(shared(var['label'][:]), 'int32') for var in y_data]
    shared_variables = shared_x + shared_y

    i = T.lscalar('index')
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size

    print('constructing Theano computational graph...')
    plc = theano.function([], cost.shape, givens={
            key: val[:batch_size]
            for key, val in zip(tensors, shared_variables)
        }
        )
    train = theano.function(
        inputs=[i],
        outputs=cost,
        updates=updates,
        givens={
            key: val[start_idx: end_idx]
            for key, val in zip(tensors, shared_variables)
        },
        name='train',
        allow_input_downcast=True,
    )

    # std_err = theano.function(
    #     inputs=[],
    #     outputs=stderrs,
    #     givens={
    #         key: val[:]
    #         for key, val in zip(tensors, shared_variables)
    #     },
    #     name='std errors',
    #     allow_input_downcast=True,
    # )

    # train model
    print(plc())
    print('training the model...')
    curves = []
    n_batches = n_samples // batch_size
    epochs = 100
    epoch = 0
    t0 = time.time()
    while epoch < epochs:
        epoch += 1
        cost = []
        for i in range(n_batches):
            cost_items = train(i)
            cost.append(cost_items)

        epoch_cost = np.sum(cost)
        curves.append((epoch, epoch_cost))
        minutes, seconds = divmod(time.time()-t0, 60.)
        hours, minutes = divmod(minutes, 60.)
        print(("epoch {0:d} loglikelihood "
              "{1:.3f} time {hh:02d}:{mm:02d}:{ss:05.2f}").format(
              epoch, epoch_cost, hh=int(hours), mm=int(minutes), ss=seconds))

        if (epoch % 5) == 0:
            print('checkpoint')
            param_values = {}
            for name, param in params.items():
                param_shp = params_shp[name]
                param_values[name] = param.eval().reshape(param_shp)
                np.savetxt('params/{}.csv'.format(name),
                           param_values[name].squeeze(),
                           fmt='%.3f', delimiter=',')

            to_file = param_values, curves
            path = 'params/epoch_{0:d}.params'.format(epoch)
            with open(path, 'wb') as f:
                pickle.dump(to_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save parameters and stderrs to .csv
    stderrs = std_err()
    for se, param, name in zip(stderrs, params.values(), params):
        v = param.eval().squeeze()
        shp = v.shape
        path = 'params/stderrs_{}.csv'.format(name)
        np.savetxt(path, se.reshape(shp), fmt='%.3f', delimiter=',')
        path = 'params/tstat_{}.csv'.format(name)
        np.savetxt(path, v / se.reshape(shp), fmt='%.3f', delimiter=',')


if __name__ == '__main__':
    dataset = SetupH5PY.load_dataset('data.h5')
    main(dataset)
