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

# CONSTANTS
VARIABLE_DTYPE_BINARY = 'binary'
VARIABLE_DTYPE_REAL = 'real'
VARIABLE_DTYPE_CATEGORY = 'category'
VARIABLE_DTYPE_INTEGER = 'integer'
DTYPE_FLOATX = theano.config.floatX


def main(data):
    # optimizer
    opt = Optimizers()

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

        beta_f = shared(beta_value, name)
        beta_params_f.append(beta_f)

        beta_params.append(T.reshape(beta_f, shp))
        beta_params_m.append(shared(mask, name + '_mask'))

        params[name] = beta_f
        params_shp[name] = shp

    # compute the utility function
    utility = 0.
    for x, b in zip(input, beta_params):
        ax = [np.arange(x.ndim)[1:], np.arange(b.ndim)[:-1]]
        utility += T.tensordot(x, b, axes=ax)

    for y, asc in zip(output, asc_params):
        utility += asc
        p_y_given_x = T.nnet.softmax(utility)
        cost = -T.sum(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

    gparams = asc_params + beta_params_f
    grads = T.grad(cost, gparams)

    # mask gradient updates
    mask = asc_params_m + beta_params_m
    for j, grad in enumerate(grads):
        grads[j] = grad * mask[j]

    # create list of updates to theano function
    updates = opt.sgd_updates(gparams, grads, lr)

    stderrs = []
    hessian = T.hessian(cost=cost, wrt=gparams)
    stderr = [T.sqrt(f) for f in [T.diag(1. / h) for h in hessian]]
    stderrs.extend(stderr)

    tensors = input + output
    shared_x = [shared(var['data'][:], borrow=True) for var in x_data]
    shared_y = [T.cast(shared(var['label'][:]), 'int32') for var in y_data]
    shared_variables = shared_x + shared_y

    i = T.lscalar('index')
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size

    print('constructing Theano computational graph...')
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

    std_err = theano.function(
        inputs=[],
        outputs=stderrs,
        givens={
            key: val[:]
            for key, val in zip(tensors, shared_variables)
        },
        name='std errors',
        allow_input_downcast=True,
    )

    # train model
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

        if (epoch % 100) == 0:
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
