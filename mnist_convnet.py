from theano import function
from nn import MLP, nnet
from optimization import adadelta, CE
from theano.tensor.nnet import relu, softmax, sigmoid, hard_sigmoid, conv
from keras.datasets import mnist

input_size = 28*28
output_size = 10


(X_train, y_train), (X_test, y_test) = mnist.load_data()

#X_train = X_train.reshape((X_train.shape[0], input_size))

#X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
y_train = y_train.flatten()


#X_test = X_test.reshape((X_test.shape[0], input_size))

#X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))

y_test = y_test.flatten()

model = nnet((3, 32, 32), output_size)

model.addConvolutionLayer(32, 4, 4)
model.addActivation(relu)
model.addConvolutionLayer(32, 4, 4)
model.addMaxPooling(2, 2)
model.addActivation(relu)
model.addConvolutionLayer(32, 4, 4)
model.addActivation(relu)
model.addConvolutionLayer(16, 4, 4)
model.addActivation(relu)

model.addFullyConnectedLayer(200)
model.addActivation(hard_sigmoid)

model.addFullyConnectedLayer(100)

model.addActivation(softmax)

model.train(X_train, y_train, X_test, y_test, 15, 100, adadelta(1e-1), CE(0.005))
model.train(X_train, y_train, X_test, y_test, 50, 100, adadelta(1e-2), CE(0.005))
#print(X_train.shape)