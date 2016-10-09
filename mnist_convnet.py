from theano import function
from nn import nnet
from optimization import adadelta, CE, sgd
from theano.tensor.nnet import relu, softmax, sigmoid, hard_sigmoid, conv
from keras.datasets import mnist

input_size = 28*28
output_size = 10


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

y_train = y_train.flatten()

y_test = y_test.flatten()

#
model = nnet((1, 28, 28))

model.addConvolutionLayer((20, 3, 3))
model.addActivation(relu)
model.addConvolutionLayer((20, 3, 3))
model.addMaxPooling((2, 2))
model.addActivation(relu)
model.addConvolutionLayer((15, 3, 3))
model.addActivation(relu)
model.addConvolutionLayer((10, 3, 3))
model.addActivation(relu)

model.addFullyConnectedLayer(200)
model.addActivation(hard_sigmoid)

model.addFullyConnectedLayer(100)

model.addActivation(softmax)

model.train(X_train, y_train, X_test, y_test, 10, 100, sgd(1e-2), CE(0.005))
model.train(X_train, y_train, X_test, y_test, 10, 100, sgd(1e-3), CE(0.005))