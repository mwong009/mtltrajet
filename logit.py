import theano
import time
import pickle
import numpy as np
import pandas as pd
import theano.tensor as T
from theano import shared
from collections import OrderedDict
from optimizers import Optimizers
from utility import SetupH5PY

# CONSTANTS
VARIABLE_DTYPE_BINARY = 'binary'
VARIABLE_DTYPE_REAL = 'real'
VARIABLE_DTYPE_CATEGORY = 'category'
VARIABLE_DTYPE_INTEGER = 'integer'
DTYPE_FLOATX = theano.config.floatX


def main(dataset):
    # optimizer
    opt = Optimizers()

    # import dataset
    n_samples = dataset.attrs['n_rows']
    lr = 1e-3
    batch_size = 128

    x = [dataset['purpose'], dataset['avg_speed'],
         dataset['duration'], dataset['trip_km'],
         dataset['n_coord'], dataset['interval'],
         dataset['dow'], dataset['startdistrict'],
         dataset['enddistrict']]

    y = [dataset['mode']]

    params = OrderedDict()
    params_shp = OrderedDict()

    output = []
    input = []
    asc_params = []
    asc_mask_params = []
    beta_flat_params = []
    beta_params = []
    beta_mask_params = []

    for var in y:
        name = var.name.strip('/')
        shp = var['data'].shape[-1]
        print('y', name, (shp,))

        tsr_output = T.ivector(name)
        asc_shp = (shp,)
        asc_name = 'asc_' + name
        mask = np.zeros(asc_shp)
        mask[:-1] = 1.
        asc_init = np.random.normal(0., 0.1, asc_shp) * mask

        asc = shared(asc_init, asc_name)
        asc_mask = shared(mask, asc_name+'_mask')

        output.append(tsr_output)
        asc_params.append(asc)
        asc_mask_params.append(asc_mask)
        params[asc_name] = asc
        params_shp[asc_name] = asc_shp

    for var in x:
        var_dtype = var.attrs['dtype']
        name = var.name.strip('/')
        shp = var['data'].shape[1:]
        print('x', name, shp)

        beta_shp = shp + asc_shp
        beta_name = 'beta_' + name
        mask = np.zeros(beta_shp)

        if len(shp) == 1:
            tsr_input = T.matrix(name)
            mask[:, :-1] = 1.
        else:
            tsr_input = T.tensor3(name)
            mask[:, :, :-1] = 1.
        mask = mask.flatten()
        beta_init = np.random.normal(0., 0.1, np.prod(beta_shp)) * mask

        beta_flat = shared(beta_init, beta_name)
        beta = T.reshape(beta_flat, beta_shp)
        beta_mask = shared(mask, beta_name+'_mask')

        input.append(tsr_input)
        beta_flat_params.append(beta_flat)
        beta_params.append(beta)
        beta_mask_params.append(beta_mask)
        params[beta_name] = beta_flat
        params_shp[beta_name] = beta_shp

    # compute the utility function
    utility = 0.
    for var, b in zip(input, beta_params):
        # utility += (rows, outs)
        if var.ndim == 2:
            utility += T.dot(var, b)
        else:
            utility += T.tensordot(var, b, [[1, 2], [0, 1]])

    for o, c in zip(output, asc_params):
        utility += c

        p_y_given_x = T.nnet.softmax(utility)
        nll = -T.sum(T.log(p_y_given_x)[T.arange(o.shape[0]), o])

    param_list = asc_params + beta_flat_params
    grads = T.grad(nll, param_list, disconnected_inputs='ignore')

    # mask gradient updates
    mask_list = asc_mask_params + beta_mask_params
    for i, (p, g, m) in enumerate(zip(param_list, grads, mask_list)):
        grads[i] = (g * m)

    updates = opt.adam_updates(param_list, grads, lr)

    sigmas = []
    hessian = T.hessian(cost=nll, wrt=param_list, disconnected_inputs='ignore')
    sigma = [T.sqrt(cr) for cr in [T.diag(1. / h) for h in hessian]]
    sigmas.extend(sigma)

    t_var = input + output
    s_var = [shared(var['data'][:], borrow=True) for var in x] + \
        [T.cast(shared(var['label'][:], borrow=True), 'int32') for var in y]

    i = T.lscalar('index')
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size

    print('constructing Theano computational graph...')
    train = theano.function(
        inputs=[i],
        outputs=nll,
        updates=updates,
        givens={
            key: val[start_idx: end_idx] for key, val in zip(t_var, s_var)},
        name='train',
        allow_input_downcast=True,
        on_unused_input='ignore'
    )

    std_err = theano.function(
        inputs=[],
        outputs=sigmas,
        givens={key: val[:] for key, val in zip(t_var, s_var)},
        name='std err',
        allow_input_downcast=True,
        on_unused_input='ignore'
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

        epoch_cost = np.asarray(cost).sum()
        curves.append((epoch, epoch_cost))
        minutes, seconds = divmod(time.time()-t0, 60.)
        hours, minutes = divmod(minutes, 60.)
        print(("epoch {0:d} loglikelihood "
              "{1:.3f} t: {hh:02d}:{mm:02d}:{ss:04.2f}").format(
              epoch, epoch_cost, hh=int(hours), mm=int(minutes), ss=seconds))

        if (epoch % 5) == 0:
            print('checkpoint')
            param_values = {}
            for param_name, param in params.items():
                param_shp = params_shp[param_name]
                param_values[param_name] = param.eval().reshape(param_shp)
                np.savetxt('params/logit_'+param_name+'.csv',
                           param_values[param_name].squeeze(),
                           fmt='%.3f', delimiter=',')

            to_file = param_values, curves
            path = 'params/{0:s}_epoch{1:d}.params'.format('logit', epoch)
            with open(path, 'wb') as f:
                pickle.dump(to_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save parameters and stderrs to .csv
    stderrs = std_err()
    for se, param, name in zip(stderrs, params.values(), params):
        v = param.eval().squeeze()
        shp = v.shape
        strfmt = 'params/'+name+'stderrs.csv'
        np.savetxt(strfmt, se.reshape(shp), fmt='%.3f', delimiter=',')
        strfmt = 'params/'+name+'tstat.csv'
        np.savetxt(strfmt, v / se.reshape(shp), fmt='%.3f', delimiter=',')


if __name__ == '__main__':
    dataset = SetupH5PY.load_dataset('data.h5')
    main(dataset)
