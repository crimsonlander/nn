from theano import function
from nn import MLP, nnet
from optimization import adadelta, CE
from theano.tensor.nnet import relu, softmax, sigmoid, hard_sigmoid, conv
from keras.datasets import cifar100

input_size = 28*28
output_size = 10


(X_train, y_train), (X_test, y_test) = cifar100.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

model = nnet((3, 32, 32))

model.addConvolutionLayer((32, 4, 4))
model.addActivation(relu)
model.addConvolutionLayer((32, 4, 4))
model.addMaxPooling((2, 2))

model.addActivation(relu)
model.addConvolutionLayer((32, 4, 4))
model.addActivation(relu)
model.addConvolutionLayer((16, 4, 4))
model.addActivation(relu)

model.addFullyConnectedLayer(200)
model.addActivation(hard_sigmoid)
model.addFullyConnectedLayer(100)
model.addActivation(softmax)

model.train(X_train, y_train, X_test, y_test, 15, 100, adadelta(1e-1), CE(0.005))
model.train(X_train, y_train, X_test, y_test, 50, 100, adadelta(1e-2), CE(0.005))