from theano import function, shared, grad
import theano.tensor as T
from nn import MLP, nnet
from optimization import adadelta, CE, MSE, momentum, sgd
from theano.tensor.nnet import relu, softmax, sigmoid, hard_sigmoid, conv
from keras.datasets import mnist
from theano.tensor.var import TensorVariable
import numpy as np

input_size = 28*28
output_size = 10


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], input_size)) / 255
y_train = y_train.flatten()

X_test = X_test.reshape((X_test.shape[0], input_size)) / 255
y_test = y_test.flatten()

model = nnet((input_size,))
model.classification = False
model.y = T.matrix(dtype='float32')

layer_size = 500

W = shared(np.random.uniform(-.1, .1, (input_size, layer_size)).astype('float32'))

model.addFullyConnectedLayer(layer_size, W)
model.addActivation(sigmoid)
model.addFullyConnectedLayer(input_size, W.T)


#f = function([model.X], error, givens=[(model.y, model.X)])
#config.tensor.cmp_sloppy=1

#model.train(X_train[:100], X_train[:100], X_test[:100], X_test[:100], 10, 100, momentum(0.001), MSE(), random_order=False)

model.train_unsupervised(X_train, X_test, 20, 100, momentum(), MSE(0.001))

'''
n_epochs = 10
batch_size = 100
optimization_function = momentum(0.01)
cost_function = MSE()

cost = cost_function(model.y, model.out, model.params)

X_train_shared = shared(X_train.astype('float32'))


N = X_train.shape[0]
n_batches = N // batch_size + (N % batch_size != 0)
index = T.iscalar()
batch_begin = index * batch_size
batch_end   = T.min(((index+1) * batch_size, N))

upd = optimization_function(model.params, cost)

optimize = function([index], [cost],givens=[(model.X, X_train_shared[batch_begin:batch_end]),(model.y, X_train_shared[batch_begin:batch_end])],updates=upd)

for j in range(n_epochs):
    for i in range(n_batches):
        print(optimize(i))


    def train_unsupervised(model, X_train, X_valid,
                           n_epochs, batch_size,
                           optimization_function,
                           cost_function):
        cost = cost_function(model.y, model.out, model.params)

        X_train_shared = shared(X_train.astype('float32'))


        N = X_train.shape[0]
        n_batches = N // batch_size + (N % batch_size != 0)
        index = T.iscalar()
        batch_begin = index * batch_size
        batch_end   = T.min(((index+1) * batch_size, N))

        upd = optimization_function(model.params, cost)

        optimize = function([index], [cost],givens=[(model.X, X_train_shared[batch_begin:batch_end]),(model.y, X_train_shared[batch_begin:batch_end])],updates=upd)

        for j in range(n_epochs):
            for i in range(n_batches):
                print(optimize(i))

'''