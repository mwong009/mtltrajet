import theano
import time
import sys
import os
import numpy as np
import pandas as pd
import theano.tensor as T
from collections import OrderedDict

# internal imports
from network import Network
from utility import SetupH5PY, init_tensor

# CONSTANTS
VARIABLE_DTYPE_BINARY = 'binary'
VARIABLE_DTYPE_REAL = 'real'
VARIABLE_DTYPE_CATEGORY = 'category'
VARIABLE_DTYPE_INTEGER = 'integer'
DTYPE_FLOATX = theano.config.floatX


class RBM(Network):
    ''' define the RBM toplevel '''
    def __init__(self, name, hyperparameters=OrderedDict()):
        Network.__init__(self, name, hyperparameters)
        # tensors
        self.label = []         # label tensors
        self.input, self.output = [], []    # list of tensors
        self.in_dtype, self.out_dtype = [], []    # list of str dtypes

        # parameters
        self.W_params, self.B_params = [], []  # xWh, hWy, xBy params
        self.V_params, self.U_params = [], []  # xWh, hWy params
        self.hbias, self.vbias, self.cbias = [], [], []  # bias

        # flattened version
        self.W_params_f, self.B_params_f = [], []  # xWh, hWy, xBy params
        self.V_params_f, self.U_params_f = [], []  # xWh, hWy params
        self.vbias_f, self.cbias_f = [], []

        # parameter masks
        self.B_params_m, self.U_params_m = [], []  # list of the Uh mask
        self.cbias_m = []

    def add_latent(self, name='hbias'):
        """
        add_latent func

        Parameters
        ----------
        name : `str`, optional
            Name of hidden node e.g. `'hbias'`
        shp_hidden : `tuple`, optional
            Size of the hidden units

        Updates
        -------
        self.hbias[] : sequence of `theano.shared()`
        self.params[name] : OrderedDict of `theano.shared()`
        """
        try:
            shp_hidden = self.hyperparameters['n_hidden']
        except KeyError as e:
            print("hidden unit shape not defined!")

        if name in self.model_values.keys():
            value = self.model_values[name]
        else:
            value = np.zeros(shp_hidden, DTYPE_FLOATX)

        hbias = theano.shared(value, name=name)

        self.hbias.append(hbias)
        self.params[name] = hbias
        self.params_shp[name] = shp_hidden

    def add_node(self, var_dtype, name, shp_visible):
        """
        add_node func

        Parameters
        ----------
        var_dtype : `str`
            Type of variables e.g. 'binary', 'category',
            see hyperparameters for more information
        name : `str`
            Name of visible node e.g. 'age'
        shp_visible : `tuple`, optional
            Size of the visible units

        Updates
        -------
        self.input[] : sequence of `T.tensor3()`\n
        self.in_dtype[] : sequence of `str`\n
        self.W_params[] : sequence of `theano.shared()`\n
        self.vbias[] : sequence of `theano.shared()`\n
        self.params['x_'+name] : OrderedDict of `theano.shared()`\n
        """
        self.hyperparameters['shapes'][name] = shp_visible

        shp_hidden = self.hyperparameters['n_hidden']
        size = shp_visible + shp_hidden

        # create the tensor symbolic variables
        tsr_variable = init_tensor(shp_visible, name)

        # create the tensor shared variables
        if 'W_' + name in self.model_values.keys():
            value = self.model_values['W_'+name]
        else:
            value = np.random.normal(0., 0.1, np.prod(size))

        W_f = theano.shared(value, 'W_'+name)
        W = T.reshape(W_f, size)

        size = shp_visible
        if 'vbias_' + name in self.model_values.keys():
            value = self.model_values['vbias_'+name]
        else:
            value = np.random.normal(0, 0.1, np.prod(shp_visible))

        vbias_f = theano.shared(value, 'vbias_'+name)
        vbias = T.reshape(vbias_f, size)

        self.input.append(tsr_variable)
        self.in_dtype.append(var_dtype)
        self.W_params.append(W)
        self.W_params_f.append(W_f)
        self.V_params.append(W)
        self.V_params_f.append(W_f)
        self.vbias.append(vbias)
        self.vbias_f.append(vbias_f)

        self.params['W_' + name] = W_f
        self.params['vbias_' + name] = vbias_f
        self.params_shp['W_' + name] = shp_visible + shp_hidden
        self.params_shp['vbias_' + name] = shp_visible

    def add_connection_to(self, var_dtype, name, shp_output):
        """
        add_connection_to func

        Parameters
        ----------
        var_dtype : `str`
            Type of variables e.g. `'binary'`, `'category'`, see
            hyperparameters for more information
        name : `str`
            Name of visible node e.g. `'mode_prime'`
        shp_output : `tuple`, optional
            Size of the visible units

        Updates
        -------
        self.output[] : sequence of `T.matrix()`
        self.W_params[] : sequence of `theano.shared()`
        self.cbias[] : sequence of `theano.shared()`
        self.B_params[] : sequence of `theano.shared()`
        self.params[] : sequence of `theano.shared()`
        """
        self.hyperparameters['shapes'][name] = shp_output
        shp_hidden = self.hyperparameters['n_hidden']

        # create the tensor symbolic variables
        tsr_variable = init_tensor(shp_output, name)
        tsr_label = T.ivector(name + '_label')

        # create logit mask for W
        size = shp_output + shp_hidden
        mask = np.ones(size)
        mask[..., -1] = 0.
        mask = mask.flatten()

        # create the tensor shared variables W
        w_name = 'W_' + name
        size = shp_output + shp_hidden
        if w_name in self.model_values.keys():
            value = self.model_values[w_name]
        else:
            value = np.random.normal(0., 0.1, np.prod(size)) * mask

        W_f = theano.shared(value, w_name)
        W_m = theano.shared(mask, w_name+'_mask')
        W = T.reshape(W_f, size)

        # create logit mask for cbias
        mask = np.ones(shp_output)
        mask[..., -1] = 0.
        mask = mask.flatten()

        # create the tensor shared variables cbias
        c_name = 'cbias_' + name
        print('cbias', name, shp_output)
        if c_name in self.model_values.keys():
            value = self.model_values[c_name]
        else:
            value = np.zeros(np.prod(shp_output), DTYPE_FLOATX) * mask

        cbias_f = theano.shared(value, c_name)
        cbias_m = theano.shared(mask, c_name+'_mask')
        cbias = T.reshape(cbias_f, shp_output)

        self.output.append(tsr_variable)
        self.out_dtype.append(var_dtype)
        self.label.append(tsr_label)
        self.W_params.append(W)
        self.U_params.append(W)
        self.cbias.append(cbias)
        self.W_params_f.append(W_f)
        self.U_params_f.append(W_f)
        self.U_params_m.append(W_m)
        self.cbias_f.append(cbias_f)
        self.cbias_m.append(cbias_m)

        self.params['W_' + name] = W_f
        self.params[c_name] = cbias_f
        self.params_shp['W_' + name] = shp_output + shp_hidden
        self.params_shp[c_name] = shp_output

        # condtional RBM connection (B weights)
        for node in self.input:
            var_name = node.name
            shp_visible = self.hyperparameters['shapes'][var_name]

            # create logit mask for B
            size = shp_visible + shp_output
            mask = np.ones(size)
            mask[..., -1] = 0.
            mask = mask.flatten()

            # create the tensor shared variables B
            b_name = 'B_' + var_name + '_' + name
            print('B', var_name, size)
            if b_name in self.model_values.keys():
                value = self.model_values[b_name]
            else:
                value = np.random.normal(0., 0.1, np.prod(size)) * mask

            B_f = theano.shared(value, b_name)
            B_m = theano.shared(mask, b_name+'_mask')
            B = T.reshape(B_f, size)

            self.B_params.append(B)
            self.B_params_f.append(B_f)
            self.B_params_m.append(B_m)

            self.params[b_name] = B_f
            self.params_shp[b_name] = shp_visible + shp_output

    def free_energy(self, input=None, utility=0):
        """
        Free energy function

        Parameters
        ----------
        self : RBM class object

        input : `[T.tensors]`, optional
            Used when calculating free energy of gibbs chain sampling

        Returns
        -------
        F(y, x) :
            Scalar value of the generative model free energy

        :math:
        `F(y, x, h) = -(xWh + yWh + vbias*x + hbias*h + cbias*y)`\n
        `    wx_b = xW + yW + hbias`\n
        `  F(y, x) = -{vbias*x + cbias*y + sum_k[ln(1+exp(wx_b))]}`\n

        """
        # if given an input. e.g. free_energy(chain_end)
        if input is None:
            visibles = self.input
        else:
            visibles = input

        # collect parameters
        dtypes = self.in_dtype
        hbias = self.hbias[0]
        vbiases = self.vbias
        W_params = self.W_params

        # input shapes as (rows, items, cats) or (rows, outs)
        # weight shapes as (items, cats, hiddens) or (outs, hiddens)
        # bias shapes as (items, cats) or (outs,)
        wx_hbias = hbias
        for dtype, v, W, vbias in zip(dtypes, visibles, W_params, vbiases):
            # vbias_x: (rows,)
            if dtype == VARIABLE_DTYPE_CATEGORY:
                if vbias.ndim > 1:
                    vbias_x = T.tensordot(v, vbias, axes=[[1, 2], [0, 1]])
                else:
                    vbias_x = T.tensordot(v, vbias, axes=[[1], [0]])
            else:
                if vbias.ndim > 1:
                    vbias = vbias.dimshuffle('x', 0, 1)
                    vbias_x = T.sum(T.sqr(v - vbias) / 2., axis=(1, 2))
                else:
                    vbias = vbias.dimshuffle('x', 0)
                    vbias_x = T.sum(T.sqr(v - vbias) / 2., axis=1)
            # utility: (rows,)
            utility += vbias_x

            # wx: (rows, hiddens)
            if W.ndim == 2:
                wx = T.dot(v, W)
            else:
                wx = T.tensordot(v, W, axes=[[1, 2], [0, 1]])
            # wx_hbias: (rows, hiddens)
            wx_hbias += wx

        # sums over hidden axis --> (rows,)
        return - utility - T.sum(T.log(1 + T.exp(wx_hbias)), axis=1)

    def discriminative_free_energy(self, input=None):
        """
        Discriminative_free_energy function
            The correct output is p(y|x)

        Parameters
        ----------
        self : RBM class object

        input : `[T.tensors]`, optional
            Used when calculating free energy of gibbs chain sampling

        Returns
        -------
        F(y|x) :
            A `list[]` of vectors of the discriminative model free energy
            for each output node. Negative loglikelihood can be used as the
            objective function.

        Notes
        -----
        The free energy for the discriminative model is computed as:

        :math:
        `F(y,x,h) = (xWh + yWh + yBx + vbias*x + hbias*h + cbias*y)`\n
        `    wx_b = xW_{ik} + yW_{jk} + hbias`\n
        `  F(y,x) = {cbias*y + yBx + sum_k[ln(1+exp(wx_b))]}`\n
        `  F(y|x) = {cbias + Bx + sum_k[ln(1+exp(wx_b)]}`\n
        `  F(y|x) = {cbias + Bx + hbias + yWh}`\n

        :params: used are W^1, W^2, B, c, h biases

        """
        # amend input if given an input. e.g. free_energy(chain_end)
        if input is None:
            visibles = self.input
        else:
            visibles = input

        # collect parameters
        dtypes = self.in_dtype
        hbias = self.hbias[0]
        vbiases = self.vbias
        cbiases = self.cbias
        xWh_params = self.V_params
        hWy_params = self.U_params
        B_params = self.B_params
        B_params_m = self.B_params_m
        cbiases_m = self.cbias_m

        # rebroadcast (hiddens,): broadcast(T, F, T) --> ('x', 0, 'x')
        wx_b = hbias.dimshuffle('x', 0, 'x')
        utility = []

        for cbias in cbiases:
            # (items, outs) --> ('x', outs)
            # utility = [cbias,...]  ('x', outs)
            cbias = T.sum(cbias, axis=0)
            u = cbias.dimshuffle('x', 0)
            utility.append(u)

        # loop over all input nodes
        # x : input variables
        # W, B : weights
        # a : input biases
        for x, xWh, B, vbias, dt in zip(visibles, xWh_params, B_params,
                                        vbiases, dtypes):
            # matrix dot product between input variables and hidden units
            # xw = xW_{ik} : (rows, hiddens)
            # wx_b = xW_{ik} + hbias : (rows, hiddens) --> (rows, hids, 'x')
            if xWh.ndim == 2:
                xw = T.dot(x, xWh)
                wx_b += xw.dimshuffle(0, 1, 'x')
            else:
                xw = T.tensordot(x, xWh, axes=[[1, 2], [0, 1]])
                wx_b += xw.dimshuffle(0, 1, 'x')

            # loop over all output nodes
            # hWy : weights (items, outs, hiddens)
            for i, hWy in enumerate(hWy_params):
                # wx_b = W_{jk} + W_{jk} + hbias : (rows, hiddens, outs)
                hWy = T.sum(hWy, axis=0)
                wx_b += hWy.dimshuffle('x', 1, 0)
                # xB : (rows, items, cats) . (items, cats, items, outs)
                # utility[i] = cbias + Bx : (rows, outs)
                # utility[i] = cbias + Bx : (rows, outs)
                if x.ndim > 2:
                    utility[i] += T.tensordot(x, B, axes=[[1, 2], [0, 1]])
                else:
                    utility[i] += T.tensordot(x, B, axes=[[1], [0]])

        # sum over hiddens axis
        # sum_k \ln(1+\exp(wx_b)) : (rows, hiddens, outs) -- > (rows, outs)
        entropy = T.sum(T.log(1 + T.exp(wx_b)), axis=1)

        # add entropy to each expected utility term
        # -F(y|x)  (rows, outs)
        energy = []
        for u in utility:
            energy.append(-u.squeeze()-entropy)

        return energy

    def sample_h_given_v(self, v0_samples, vtype='xy'):
        """
        sample_h_given_v func
            Binomial hidden units

        Parameters
        ----------
        v0_samples : `[T.tensors]`
            theano Tensor variable

        Returns
        -------
        h1_preactivation : `scalar` (-inf, inf)
            preactivation function e.g. logit utility func
        h1_means : `scalar` (0, 1)
            sigmoid activation
        h1_samples : `integer` 0 or 1
            binary samples
        """
        # prop up
        if vtype == 'xy':
            W_params = self.W_params
        elif vtype == 'x':
            W_params = self.V_params
        elif vtype == 'y':
            W_params = self.U_params
        else:
            print('error')

        hbias = self.hbias
        h1_preactivation = self.propup(v0_samples, W_params, hbias[0])

        # h ~ p(h|v0_sample)
        h1_means = T.nnet.sigmoid(h1_preactivation)
        h1_samples = self.theano_rng.binomial(
            size=h1_means.shape,
            p=h1_means,
            dtype=DTYPE_FLOATX
        )

        return h1_preactivation, h1_means, h1_samples

    def propup(self, samples, weights, bias):

        preactivation = bias
        # (rows, items, cats), (items, cats, hiddens)
        # (rows, outs), (outs, hiddens)
        for v, W, in zip(samples, weights):
            if W.ndim == 2:
                preactivation += T.dot(v, W)
            else:
                preactivation += T.tensordot(v, W, axes=[[1, 2], [0, 1]])

        return preactivation

    def sample_v_given_h(self, h0_samples, vtype='xy'):
        """
        sample_v_given_h func
            Binomial hidden units

        Parameters
        ----------
        h0_samples : `[T.tensors]`
            theano Tensor variable

        Returns
        -------
        v1_preactivation : `[scalar]` (-inf, inf)
            sequence of preactivation function e.g. logit utility func
        v1_means : `[scalar]` (0, 1)
            sequence of sigmoid activation
        v1_samples : `[binary]` or `[integer]` or `[float32]` or `[array[j]]`
            visible unit samples
        """
        # prop down
        if vtype == 'xy':
            W_params = self.W_params
            bias = self.vbias + self.cbias
            dtypes = self.in_dtype + self.out_dtype

        elif vtype == 'x':
            W_params = self.V_params
            bias = self.vbias
            dtypes = self.in_dtype

        elif vtype == 'y':
            W_params = self.U_params
            bias = self.cbias
            dtypes = self.out_dtype
        else:
            print('error')

        v1_preactivation = self.propdown(h0_samples, W_params, bias)

        # v ~ p(v|h0_sample)
        v1_means = []
        v1_samples = []
        for v1, dtype in zip(v1_preactivation, dtypes):
            if dtype == VARIABLE_DTYPE_BINARY:
                v1_mean = T.nnet.sigmoid(v1)
                v1_sample = self.theano_rng.binomial(
                    size=v1.shape,
                    p=v1_mean,
                    dtype=DTYPE_FLOATX
                )

            elif dtype == VARIABLE_DTYPE_CATEGORY:
                # softmax temperature value \tau (default=1)
                tau = 0.7
                uniform = self.theano_rng.uniform(
                    size=v1.shape,
                    low=1e-10,
                    high=1.0,
                    dtype=DTYPE_FLOATX
                )
                gumbel = -T.log(-T.log(uniform))
                # reshape softmax tensors to 2D matrix
                if v1.ndim == 3:
                    (d1, d2, d3) = v1.shape
                    logit = (v1 + gumbel).reshape((d1 * d2, d3))
                    v1_mean = T.nnet.softmax(logit / tau)
                    # reshape back into original dimensions
                    v1_mean = v1_mean.reshape((d1, d2, d3))
                else:
                    logit = (v1 + gumbel)
                    # (rows, items, cats)
                    v1_mean = T.nnet.softmax(logit / tau)
                v1_sample = v1_mean

            elif dtype == VARIABLE_DTYPE_REAL:
                v1_std = T.nnet.sigmoid(v1)
                normal_sample = self.theano_rng.normal(
                    size=v1.shape,  # (rows, items, cats)
                    avg=v1,
                    std=v1_std,
                    dtype=DTYPE_FLOATX
                )
                v1_sample = normal_sample

            elif dtype == VARIABLE_DTYPE_INTEGER:
                v1_std = T.nnet.sigmoid(v1)
                normal_sample = self.theano_rng.normal(
                    size=v1.shape,
                    avg=v1,
                    std=v1_std,
                    dtype=DTYPE_FLOATX
                )
                v1_sample = T.nnet.softplus(normal_sample)

            else:
                raise NotImplementedError

            v1_means.append(v1_mean)
            v1_samples.append(v1_sample)

        return v1_preactivation, v1_means, v1_samples

    def propdown(self, samples, weights, bias):

        preactivation = []
        # (rows, hiddens), (items, cats, hiddens) --> dimshuffle(0, 2, 1)
        # (rows, hiddens), (outs, hiddens) --> dimshuffle(1, 0)
        for W, b in zip(weights, bias):
            if W.ndim == 2:
                W = W.dimshuffle(1, 0)
            else:
                W = W.dimshuffle(0, 2, 1)
            preactivation.append(T.dot(samples, W) + b)

        return preactivation

    def gibbs_hvh(self, h0_samples):
        v1_pre, v1_means, v1_samples = self.sample_v_given_h(h0_samples)
        h1_pre, h1_means, h1_samples = self.sample_h_given_v(v1_samples)

        return v1_pre + v1_means + v1_samples + \
            [h1_pre] + [h1_means] + [h1_samples]

    def gibbs_vhv(self, *v0_samples):
        h1_pre, h1_means, h1_samples = self.sample_h_given_v(v0_samples)
        v1_pre, v1_means, v1_samples = self.sample_v_given_h(h1_samples)

        return [h1_pre] + [h1_means] + [h1_samples] + \
            v1_pre + v1_means + v1_samples

    def get_generative_cost_updates(self, k=1, lr=1e-3):
        """
        get_generative_cost_updates func
            updates weights for W^(1), W^(2), a, c and d
        """
        # prepare visible samples from x input and y outputs
        v0_samples = self.input + self.output
        labels = self.label

        # perform positive Gibbs sampling phase
        # one step Gibbs sampling p(h|v1,v2,...) = p(h|v1)+p(h|v2)+...
        h1_pre, h1_means, h1_samples = self.sample_h_given_v(v0_samples)

        # start of Gibbs sampling chain
        # we only want the samples generated from the Gibbs sampling phase
        chain_start = h1_samples
        scan_out = 3 * len(v0_samples) * [None] + [None, None, chain_start]

        # theano scan function to loop over all Gibbs steps k
        # [v1_pre[], v1_means[], v1_samples[], h1_pre, h1_means, h1_samples]
        # outputs are given by outputs_info
        # [[t,t+1,t+2,...], [t,t+1,t+2,...], ], gibbs_updates
        # NOTE: scan returns a dictionary of updates
        gibbs_output, gibbs_updates = theano.scan(
            fn=self.gibbs_hvh,
            outputs_info=scan_out,
            n_steps=k,
            name='gibbs_hvh'
        )

        # note that we only need the visible samples at the end of the chain
        chain_end = []
        for output in gibbs_output:
            chain_end.append(output[-1])
        gibbs_pre = chain_end[:len(v0_samples)]
        gibbs_means = chain_end[len(v0_samples): 2 * len(v0_samples)]
        gibbs_samples = chain_end[2 * len(v0_samples): 3 * len(v0_samples)]

        # calculate the model cost
        ginitial_cost = T.mean(self.free_energy(self.input))
        gfinal_cost = T.mean(self.free_energy(gibbs_samples[:len(self.input)]))
        gcost = (ginitial_cost - gfinal_cost) * self.hyperparameters['alpha']

        dinitial_cost = self.discriminative_free_energy()
        dfinal_cost = self.discriminative_free_energy(gibbs_samples)
        dgcost = (T.mean(dinitial_cost) - T.mean(dfinal_cost))

        # conditonal probability
        dcost = 0
        for i, (logit, label) in enumerate(zip(dinitial_cost, labels)):
            p_y_given_x = T.nnet.softmax(logit)
            dcost += self.get_loglikelihood(p_y_given_x, label)
            pred = T.argmax(p_y_given_x, axis=-1)
            errors = T.neq(pred, label)

        # calculate the gradients
        gen_params = self.vbias_f + self.V_params_f + self.hbias
        gen_grads = T.grad(cost=gcost, wrt=gen_params,
                           consider_constant=gibbs_samples,
                           disconnected_inputs='ignore')

        # update Gibbs chain with update expressions from updates list[]
        gen_updates = self.update_opt(gen_params, gen_grads, lr)
        for variable, expression in gen_updates:
            gibbs_updates[variable] = expression

        # calculate the gradients
        disgen_params = self.B_params_f + self.U_params_f + self.cbias_f
        disgen_grads = T.grad(cost=dgcost+dcost, wrt=disgen_params,
                              consider_constant=gibbs_samples,
                              disconnected_inputs='ignore')

        # update Gibbs chain with update expressions from updates list[]
        disgen_updates = self.update_opt(disgen_params, disgen_grads, lr)
        for variable, expression in disgen_updates:
            gibbs_updates[variable] = expression

        # pseudo loglikelihood to track the quality of the hidden units
        # on input variables ONLY
        monitoring_cost = self.pseudo_loglikelihood(
            inputs=self.input,
            preactivation=gibbs_pre[:len(self.input)])

        return monitoring_cost, dcost, errors, gibbs_updates, [
               dinitial_cost, dfinal_cost]

    def get_v_samples(self, k):
        # prepare visible samples from input
        chain_start = self.input + self.output
        _, _, h0_samples = self.sample_h_given_v(chain_start)
        scan_out = 3*len(chain_start)*[None] + [None, None, h0_samples]

        # theano scan function to loop over all Gibbs steps k
        # [h1_pre, h1_means, h1_samples, v1_pre[], v1_means[], v1_samples[]]
        # outputs are given by outputs_info
        # [[t,t+1,t+2,...], [t,t+1,t+2,...], ], gibbs_updates
        # NOTE: scan returns a dictionary of updates
        gibbs_output, gibbs_updates = theano.scan(
            fn=self.gibbs_hvh,
            outputs_info=scan_out,
            n_steps=k,
            name='gibbs_sampling'
        )

        # # note that we only need the visible samples at the end of the chain
        chain_end = []
        for output in gibbs_output:
            chain_end.append(output[-1])
        gibbs_samples = chain_end[2*len(chain_start):3*len(chain_start)]

        return gibbs_samples, gibbs_updates

    def pseudo_loglikelihood(self, inputs, preactivation):
        """
        pseudo_loglikelihood func
            Function to calculate the (pseudo) neg loglikelihood

        Parameters
        ----------
        inputs : `[T.tensors]`
            list of input tensors
        preactivation : `[T.shared]`
            list of precomputed "logits"

        Returns
        -------
        pll : `scalar`
            value of the pseudo log likelihood
        """
        dtypes = self.in_dtype
        epsilon = 1e-10  # small value to prevent log(0.)
        cross_entropy = 0
        mse_r = 0
        mse_i = 0
        for input, v1, dtype in zip(inputs, preactivation, dtypes):
            if dtype == VARIABLE_DTYPE_BINARY:
                cross_entropy -= T.mean(T.sum(
                    input * T.log(T.nnet.sigmoid(v1))), axis=1
                )

            elif dtype == VARIABLE_DTYPE_CATEGORY:
                tau = 0.7
                (d1, d2, d3) = v1.shape
                v1_mean = T.nnet.softmax(v1.reshape((d1 * d2, d3))/tau)
                # reshape back into original dimensions
                v1_mean = v1_mean.reshape((d1, d2, d3))
                cross_entropy -= T.mean(
                    T.sum(input * T.log(v1_mean + epsilon) +
                          (1 - input) * T.log(1 - v1_mean + epsilon), axis=2)
                )

            elif dtype == VARIABLE_DTYPE_REAL:
                mse_r += T.mean(T.sqr(input - v1))

            elif dtype == VARIABLE_DTYPE_INTEGER:
                v = T.nnet.softplus(v1)
                mse_i += T.mean(T.sqr(input - v))

            else:
                raise NotImplementedError

        return [cross_entropy, mse_r, mse_i]

    def initialize(self, x, y, x_valid, y_valid):
        """
        initialize func
        # TODO

        Parameters
        ----------
        """
        print('building theano computational graphs...')
        self.add_latent()

        for item in x:
            print('x', item.name.strip('/'), item['data'].shape[1:])
            self.add_node(
                var_dtype=item.attrs['dtype'],
                name=item.name.strip('/'),
                shp_visible=item['data'].shape[1:]
            )
        for item in y:
            print('y', item.name.strip('/'), item['data'].shape[1:])
            self.add_connection_to(
                var_dtype=item.attrs['dtype'],
                name=item.name.strip('/'),
                shp_output=item['data'].shape[1:]
            )

        lr = self.hyperparameters['learning_rate']
        k = self.hyperparameters['gibbs_steps']
        batch_size = self.hyperparameters['batch_size']
        n_samples = self.hyperparameters['n_samples']
        n_slice = self.hyperparameters['slice']

        cost, dcost, errors, gibbs_updates, [
            initial_cost, final_cost] = self.get_generative_cost_updates(k, lr)

        gibbs_sampling_steps = T.lscalar('steps')
        vsamples, vsamples_updates = self.get_v_samples(gibbs_sampling_steps)

        tensor_inputs = self.input + self.output + self.label
        tensor_outputs = cost + [dcost]
        tensor_updates = gibbs_updates
        # tensor_updates = gibbs_updates.update(updates)

        shared_inputs = [
            theano.shared(item['data'][:],
                          borrow=True) for item in x] \
            + [theano.shared(item['data'][:],
                             borrow=True) for item in y] \
            + [T.cast(theano.shared(item['label'][:],
                                    borrow=True), 'int32') for item in y]

        shared_inputs_valid = [
            theano.shared(item['data'][:],
                          borrow=True) for item in x_valid] \
            + [theano.shared(item['data'][:],
                             borrow=True) for item in y_valid] \
            + [T.cast(theano.shared(item['label'][:],
                                    borrow=True), 'int32') for item in y_valid]

        i = T.lscalar('index')
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        print('constructing Theano computational graph...')
        self.train = theano.function(
            inputs=[i],
            outputs=tensor_outputs,
            updates=tensor_updates,
            givens={
                key: val[start_idx: end_idx]
                for key, val in zip(tensor_inputs, shared_inputs)},
            name='train',
            allow_input_downcast=True,
            on_unused_input='ignore'
        )

        self.sample = theano.function(
            inputs=[gibbs_sampling_steps],
            outputs=vsamples,
            updates=vsamples_updates,
            givens={
                key: val[:]
                for key, val in zip(tensor_inputs, shared_inputs_valid)},
            name='sample',
            allow_input_downcast=True,
            on_unused_input='ignore'
        )

        self.validate = theano.function(
            inputs=[i],
            outputs=errors,
            givens={
                key: val[start_idx: end_idx]
                for key, val in zip(tensor_inputs, shared_inputs)},
            name='validate',
            allow_input_downcast=True,
            on_unused_input='ignore'
        )

        logits = self.discriminative_free_energy()
        sigmas = []
        eval_params = self.B_params_f + self.cbias_f
        for i, logit in enumerate(logits):
            p_y_given_x = T.nnet.softmax(logit)
            # calculate the Hessians
            hessian = T.hessian(
                cost=self.get_loglikelihood(p_y_given_x, self.label[i]),
                wrt=eval_params,
                disconnected_inputs='ignore'
            )
            sigma = [T.sqrt(cr) for cr in [T.diag(1. / h) for h in hessian]]
            sigmas.extend(sigma)

        self.std_err = theano.function(
            inputs=[],
            outputs=sigmas,
            name='std err',
            givens={
                key: val[:]
                for key, val in zip(tensor_inputs, shared_inputs)
            },
            allow_input_downcast=True,
            on_unused_input='ignore'
        )

    def checkpoint(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        # checkpoint parameters
        params = [p for p in self.B_params + self.cbias]
        param_names = [p.name for p in self.B_params_f + self.cbias_f]
        for param, name in zip(params, param_names):
            filepath = path + name + '_' + self.name + '.csv'
            p = (param.eval()).squeeze()
            with open(filepath, 'w+') as f:
                np.savetxt(f, p, fmt='%.3f', delimiter=',')

    def final_checkpoint(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        stderrs = self.std_err()
        params = [p for p in self.B_params + self.cbias]
        param_names = [p.name for p in self.B_params_f + self.cbias_f]
        for se, param, name in zip(stderrs, params, param_names):
            v = (param.eval()).squeeze()
            shp = v.shape
            with open(path+'stderrs_'+name+'_'+self.name+'.csv') as f:
                np.savetxt(f, se.reshape(shp), fmt='%.3f', delimiter=',')
            with open(path+'tstats_'+name+'_'+self.name+'.csv') as f:
                np.savetxt(f, v / se.reshape(shp), fmt='%.3f', delimiter=',')


def save_samples(path, name, x, y, samples, steps, i):
    filepath = '{0:s}{1:s}_{2:d}draws_epoch{3:d}.csv'.format(
        path, name, steps, i)
    df = pd.DataFrame()
    if not os.path.isdir(path):
        os.mkdir(path)
    for item in (x + y):
        item_name = item.name.strip('/')
        df[item_name] = ''
    with open(filepath, 'w+') as f:
        df.to_csv(f, index=False)

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

    with open(filepath, 'a+') as f:
        df.to_csv(f, header=False, index=False)


def get_time(t0):
    minutes, seconds = divmod(time.time() - t0, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds


def main(rbm, h5py_dataset, valid_dataset, epochs):
    n_samples = h5py_dataset.attrs['n_rows']
    rbm.hyperparameters['n_samples'] = n_samples
    for key, value in rbm.hyperparameters.items():
        if isinstance(value, int) or isinstance(value, float):
            print(key, value)

    # define the variables to use
    x, x_valid = [], []
    for variable in ['purpose', 'avg_speed', 'duration', 'trip_km', 'n_coord',
                     'interval', 'dow', 'startdistrict', 'enddistrict']:
        x.append(h5py_dataset[variable])
        x_valid.append(valid_dataset[variable])
    y = [h5py_dataset['mode']]
    y_valid = [valid_dataset['mode']]

    # load the dataset into the model
    rbm.initialize(x, y, x_valid, y_valid)

    batch_size = rbm.hyperparameters['batch_size']
    n_slice = rbm.hyperparameters['slice']
    n_batches = int(n_slice*n_samples) // batch_size

    print('training the model...')
    epoch = 0
    t0 = time.time()
    while epoch < epochs:
        epoch += 1
        batch_cost, batch_error = [], []
        for i in range(n_batches):
            cost = rbm.train(i)
            batch_cost.append(cost)
            if i >= (0.9 * n_batches):
                error = rbm.validate(i)
                batch_error.append(error)

        epoch_cost = np.asarray(batch_cost).sum(axis=0)
        epoch_error = np.asarray(batch_error).mean() * 100
        hours, minutes, seconds = get_time(t0)
        print(
            ("[epoch {0:d}] {hh:02d}h{mm:02d}m{ss:04.1f}s\n"
             "error: {1:.2f}% entropy: {2:.2f} mse cost: [{3:.2f}, {4:.2f}]\n"
             "loglikelihood: {5:.2f}").format(
                epoch, epoch_error, *epoch_cost,
                hh=int(hours), mm=int(minutes), ss=seconds
            )
        )
        for i, (key, curve) in enumerate(rbm.monitoring_curves.items()):
            stats = epoch_cost.tolist() + [epoch_error]
            rbm.monitoring_curves[key].append((epoch, stats[i]))

        if (epoch % 5) == 0:
            rbm.save_params(epoch)
            rbm.plot_curves()

        steps = 100
        print('...generating samples')
        gen_samples = rbm.sample(steps)
        save_samples('samples/', rbm.name, x_valid, y_valid, gen_samples,
                     steps, epoch)
        rbm.checkpoint('params/')
    rbm.final_checkpoint('params/')

    print('train complete')

net = {
    'debug': True,
    'n_hidden': (5,),
    'seed': 4567,
    'batch_size': 128,
    'variable_dtypes': [VARIABLE_DTYPE_BINARY,
                        VARIABLE_DTYPE_REAL,
                        VARIABLE_DTYPE_CATEGORY],
    'noisy_rectifier': True,
    'learning_rate': 1e-3,
    'gibbs_steps': 1,
    'shapes': {},
    'amsgrad': True,
    'alpha': 0.01,
    'slice': 1.
}

if __name__ == '__main__':
    data = SetupH5PY.load_dataset('data.h5')
    data_valid = SetupH5PY.load_dataset('data_valid.h5')
    model = RBM('net5', net)
    main(model, data, data_valid, epochs=100)
    del(model)

    for i in np.arange(10, 105, step=5):
        net['n_hidden'] = (i,)
        model = RBM('net'+str(i), net)
        main(model, data, data_valid, epochs=100)
        del(model)
