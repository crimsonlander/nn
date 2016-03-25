from theano import function
from nn import MLP, nnet
from optimization import adadelta, CE
from theano.tensor.nnet import relu, softmax, sigmoid, hard_sigmoid, conv
from keras.datasets import mnist

input_size = 28*28
output_size = 10


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], input_size))
y_train = y_train.flatten()

X_test = X_test.reshape((X_test.shape[0], input_size))
y_test = y_test.flatten()

model = nnet((input_size,))

model.addFullyConnectedLayer(400)
model.addActivation(hard_sigmoid)
model.addFullyConnectedLayer(200)
model.addActivation(hard_sigmoid)
model.addFullyConnectedLayer(10)
model.addActivation(softmax)

model.train(X_train, y_train, X_test, y_test, 10, 100, adadelta(1e-1), CE(0.005), random_order=False)
#model.train(X_train, y_train, X_test, y_test, 10, 100, adadelta(1e-2), CE(0.005))