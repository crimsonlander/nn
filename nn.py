# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:08:45 2016

@author: denis
"""

import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import relu, softmax, sigmoid, hard_sigmoid, conv
from theano.tensor.signal.pool import pool_2d

from model import Model
from optimization import momentum, adadelta, CE
from theano import shared, function


class nnet(Model):
    def addConvolutionLayer(self, filter_shape, W=None, b=None):
        tensor_shape = (filter_shape[0], self.out_shape[0], filter_shape[1], filter_shape[2])
        bound = (1. / (self.out_shape[0] * filter_shape[1] * filter_shape[2])) ** 0.5

        if not W:
            W = shared(np.random.uniform(-bound,  bound, tensor_shape).astype('float32'))
        elif isinstance(W, np.ndarray):
            W = shared(W.astype('float32'))

        if not b:
            b = shared(np.zeros(filter_shape[0], dtype='float32'))
        elif isinstance(b, np.ndarray):
            b = shared(b.astype('float32'))

        if isinstance(W, T.sharedvar.SharedVariable):
            self.params.append(W)

        if isinstance(b, T.sharedvar.SharedVariable):
            self.params.append(b)

        self.out = conv.conv2d(self.out, W) + b.dimshuffle('x', 0, 'x', 'x')

        for i in range(1, len(self.out_shape)):
            self.out_shape[i] -= filter_shape[i] - 1

        self.out_shape[0] = filter_shape[0]

    def addFullyConnectedLayer(self, layer_size, W=None, b=None):
        full_input_size = np.prod(self.out_shape)
        self.out = self.out.reshape((self.out.shape[0], full_input_size))

        if not W:
            W = shared(np.random.normal(0, 0.2, (full_input_size, layer_size)).astype('float32'),
                          name="W")
        elif isinstance(W, np.ndarray):
            W = shared(W.astype('float32'), name="b")

        if not b:
            b = shared(np.zeros(layer_size, dtype='float32'))
        elif isinstance(b, np.ndarray):
            b = shared(b.astype('float32'))

        if isinstance(W, T.sharedvar.SharedVariable):
            self.params.append(W)

        if isinstance(b, T.sharedvar.SharedVariable):
            self.params.append(b)

        self.out = T.dot(self.out, W) + b
        self.out_shape = (layer_size, )

    def addActivation(self, activation):
        self.out = activation(self.out)

    def addMaxPooling(self, patch_shape):
        self.out = pool_2d(self.out, patch_shape, False)

        for i in range(1, len(self.out_shape)):
            self.out_shape[i] = self.out_shape[i] // patch_shape[i - 1] \
                                + (self.out_shape[i] % patch_shape[i - 1] != 0)

    def addDropout(self, p):
        dropout_p = shared(p)

    def __init__(self, input_shape, model_type='classification'):
        Model.__init__(self, model_type)
        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        if len(input_shape) == 1:
            self.X = T.matrix('X')
        elif len(input_shape) == 2:
            self.X = T.tensor3('X')
        elif len(input_shape) == 3:
            self.X = T.tensor4('X')
        else:
            raise ValueError("Unrecognized input shape")

        self.out = self.X

        self.out_shape = list(input_shape)
















