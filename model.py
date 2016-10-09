from __future__ import print_function
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function, shared
from theano.tensor.var import TensorVariable
from theano.tensor.sharedvar import SharedVariable

theano.config.warn_float64 = 'raise'

from theano.compile.debugmode import DebugMode

class Model:
    def __init__(self, model_type='classification'):
        if not model_type in ('classification', 'regression'):
            raise ValueError("Incorrect classification type. Should be either 'classification' or 'regression'.")

        self.params = []
        self.X = T.matrix(dtype='float32', name='X')

        self.out = T.matrix(dtype='float32')

        self.classification = (model_type == 'classification')

        if self.classification:
            self.y = T.vector(dtype='int32', name='y')
        else:
            self.y = T.matrix(dtype='float32')

    def prediction(self):
        if self.classification:
            return T.argmax(self.out, axis=1)
        else:
            return self.out

    def error(self):
        if self.classification:
            return 1 - T.mean(T.eq(self.y, self.prediction()), dtype='float32')
        else:
            return T.mean((self.y - self.out)**2)

    def predict(self, X):
        self._predict = function([self.X], self.prediction())
        return self._predict(X)

    def train(self, X_train, y_train,
          X_valid, y_valid,
          n_epochs, batch_size,
          optimization_function,
          cost_function,
          random_order=True):

        N = X_train.shape[0]
        n_batches = N // batch_size + (N % batch_size != 0)

        no_default_upd = []
        manual_updates = []

        if not isinstance(X_train, (TensorVariable, SharedVariable)):
            X_train = shared(X_train.astype('float32'), name="X_train")
        if not isinstance(y_train, (TensorVariable, SharedVariable)):
            if self.classification:
                y_train = shared(y_train.astype('int32'), name="y_train")
            else:
                y_train = shared(y_train.astype('float32'), name="y_train")

        if not isinstance(X_valid, (TensorVariable, SharedVariable)):
            X_valid = shared(X_valid.astype('float32'), name="X_valid")
        if not isinstance(y_valid, (TensorVariable, SharedVariable)):
            if self.classification:
                y_valid = shared(y_valid.astype('int32'), name="y_valid")
            else:
                y_valid = shared(y_valid.astype('float32'), name="y_valid")

        if random_order:
            srng = RandomStreams(seed=234)
            perm = srng.permutation(n = N)

            manual_updates.append(function([], updates=[(X_train, X_train[perm]),
                                                        (y_train, y_train[perm])]))

        cost = cost_function(self.y, self.out, self.params)
        error = self.error()

        validate = function([], [cost, error],
                            givens=[(self.X, X_valid), (self.y, y_valid)])

        index = T.iscalar()
        upd = optimization_function(self.params, cost)

        batch_begin = index * batch_size
        batch_end   = T.min(((index+1) * batch_size, N))

        optimize = function([index], [cost, error],
                            givens=[(self.X, X_train[batch_begin:batch_end]),
                                    (self.y, y_train[batch_begin:batch_end])],
                            updates=upd,
                            no_default_updates=no_default_upd)

        for epoch in range(n_epochs):
            print("Epoch", epoch)
            cost_sum, error_sum = 0, 0
            print("Running batches...")
            for i in range(n_batches):
                c, a = optimize(i)
                cost_sum += c
                error_sum += a

            print("Done!")
            print("training: cost", cost_sum / n_batches, ", error", error_sum / n_batches)
            c, a = validate()
            print("validation: cost", c, ", error", a)

            for man_upd in manual_updates:
                man_upd()

    def train_unsupervised(self, X_train, X_valid,
                           n_epochs, batch_size,
                           optimization_function,
                           cost_function):
        cost = cost_function(self.y, self.out, self.params)

        X_train_shared = shared(X_train.astype('float32'))

        N = X_train.size
        n_batches = N // batch_size + (N % batch_size != 0)

        index = T.iscalar()
        batch_begin = index * batch_size
        batch_end   = T.min(((index+1) * batch_size, N))

        upd = optimization_function(self.params, cost)

        optimize = function([index], [cost],givens=[(self.X, X_train_shared[batch_begin:batch_end]),
                                                    (self.y, X_train_shared[batch_begin:batch_end])],
                            updates=upd)

        for j in range(n_epochs):
            for i in range(n_batches):
                print(optimize(i))
