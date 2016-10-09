from __future__ import print_function, division
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import function, shared
from theano.tensor.var import TensorVariable
from theano.tensor.sharedvar import SharedVariable
import numpy as np

theano.config.warn_float64 = 'raise'



from theano.compile.debugmode import DebugMode

class Model:
    def __init__(self, model_type='classification'):
        if not model_type in ('classification', 'regression'):
            raise ValueError("Incorrect classification type. Should be either "
                             "'classification' or 'regression'.")

        self.params = []
        self.X = T.matrix(dtype='float32', name='X')

        self.out = T.matrix(dtype='float32')

        self.classification = (model_type == 'classification')

        if self.classification:
            self.y = T.vector(dtype='int32', name='y')
        else:
            self.y = T.matrix(dtype='float32', name='y')

        self.rng = MRG_RandomStreams(1)
        self.drop_out_params = []
        self.turn_off_dropout = []

        self.no_default_upd = []
        self.manual_updates = []

        self.offset = shared(np.cast["float32"](0))
        self.scale = shared(np.cast["float32"](1))

        if model_type != 'classification':
            self.yScaled = (self.y - self.offset) / self.scale

        else:
            self.yScaled = self.y

    def prediction(self):
        if self.classification:
            return T.argmax(self.out, axis=1)
        else:
            return self.out

    def rescaleInput(self, offset, scale):
        self.out = (self.out - np.cast["float32"](offset))\
                   / np.cast["float32"](scale)

    def minmaxNormalization(self, X_data):
        m, M = function([], [T.min(self.out, axis=0), T.max(self.out, axis=0)],
                        givens=[(self.X, X_data)])()

        self.out = (self.out - m) / np.float32((M - m + (M-m < 1e-5)))

    def rescaleOutput(self, offset, scale):
        self.offset.set_value(np.cast["float32"](offset))
        self.scale.set_value(np.cast["float32"](scale))

    def error(self):
        if self.classification:
            return 1 - T.mean(T.eq(self.yScaled, self.prediction()), dtype='float32')
        else:
            return T.sqrt(T.mean((self.yScaled - self.out)**2))

    def predict(self, X):
        self._predict = function([self.X], self.prediction(),
                                 givens=self.turn_off_dropout, allow_input_downcast=True)
        return self._predict(X)

    def train(self, X_train, y_train,
          X_valid, y_valid,
          n_epochs, batch_size,
          optimization_function,
          cost_function,
          random_order=True):

        unsupervised = (X_train is y_train)

        if not isinstance(X_train, (TensorVariable, SharedVariable)):
            N = X_train.shape[0]
        else:
            N = function([], X_train.shape[0])()

        n_batches = N // batch_size + (N % batch_size != 0)

        if not isinstance(X_train, (TensorVariable, SharedVariable)):
            X_train = shared(X_train.astype('float32'), name="X_train")

        if not isinstance(X_valid, (TensorVariable, SharedVariable)):
            X_valid = shared(X_valid.astype('float32'), name="X_valid")

        if not unsupervised and not isinstance(y_train, (TensorVariable, SharedVariable)):
            if self.classification:
                y_train = shared(y_train.astype('int32'), name="y_train")
            else:
                y_train = shared(y_train.astype('float32'), name="y_train")

        if not unsupervised and not isinstance(y_valid, (TensorVariable, SharedVariable)):
            if self.classification:
                y_valid = shared(y_valid.astype('int32'), name="y_valid")
            else:
                y_valid = shared(y_valid.astype('float32'), name="y_valid")

        if random_order:
            perm_rng = RandomStreams(1)
            perm = perm_rng.permutation(n=N)
            if unsupervised:
                self.manual_updates.append(function([], updates=[(X_train, X_train[perm])]))
            else:
                self.manual_updates.append(function([], updates=[(X_train, X_train[perm]),
                                                                 (y_train, y_train[perm])]))

        if unsupervised:
            y_train = X_train
            y_valid = X_valid

        cost = cost_function(self.yScaled, self.out, self.params)
        error = self.error()

        validate = function([], [cost, error],
                            givens=[(self.X, X_valid), (self.y, y_valid)]
                                   + self.turn_off_dropout,
                            no_default_updates=self.no_default_upd)

        index = T.iscalar()
        upd = optimization_function(self.params, cost)

        batch_begin = index * batch_size
        batch_end   = T.min(((index+1) * batch_size, N))

        optimize = function([index], [cost, error],
                            givens=[(self.X, X_train[batch_begin:batch_end]),
                                    (self.y, y_train[batch_begin:batch_end])],
                            updates=upd,
                            no_default_updates=self.no_default_upd)

        for epoch in range(n_epochs):
            print("Epoch", epoch)
            cost_sum, error_sum = 0, 0
            print("Running batches...")
            for i in range(n_batches):
                c, a = optimize(i)
                cost_sum += c
                error_sum += a

            print("Done!")
            print("training: cost", cost_sum / float(n_batches), ", error", error_sum / float(n_batches))
            c, a = validate()
            print("validation: cost", c, ", error", a)

            for man_upd in self.manual_updates:
                man_upd()

    def train_simple(self, X_train, y_train,
                           n_epochs, batch_size,
                           optimization_function,
                           cost_function):
        cost = cost_function(self.y, self.out, self.params)
        error = self.error()

        X_train_shared = shared(X_train.astype('float32'))
        y_train_shared = shared(y_train.astype('int32'))

        N = X_train.shape[0]
        n_batches = N // batch_size + (N % batch_size != 0)

        index = T.iscalar()
        batch_begin = index * batch_size
        batch_end   = T.min(((index+1) * batch_size, N))

        upd = optimization_function(self.params, cost)

        optimize = function([index], [cost, error],
                            givens=[(self.X, X_train_shared[batch_begin:batch_end]),
                                    (self.y, y_train_shared[batch_begin:batch_end])],
                            updates=upd)

        p = T.ivector()
        permute = function([p], updates=[(X_train_shared, X_train_shared[p]),
                                         (y_train_shared, y_train_shared[p])], allow_input_downcast=True)
        for j in range(n_epochs):
            for i in range(n_batches):
                print(optimize(i))
                permute(np.random.permutation(N))