import theano as th
import theano.tensor as T
import numpy as np


def adadelta(lr = 1., p=0.9, eps=1e-4):
    def train_updates(params, cost):
        gradients = th.grad(cost, params)

        updates = []

        for i in range(len(params)):
            shape = (params[i].get_value(borrow=True)).shape
            g = gradients[i]

            Eg2_p  = th.shared(np.zeros(shape, dtype='float32'), name='Eg2_p')
            Edx_p2 = th.shared(np.zeros(shape, dtype='float32'), name='Edx_p2')

            Eg2 = p * Eg2_p + (1 - p) * g**2

            RMSg = T.sqrt(eps + Eg2)
            RMSdx_p = T.sqrt(eps + Edx_p2)

            dx = -RMSdx_p / RMSg * g

            updates.append((Eg2_p, Eg2))
            updates.append((Edx_p2, p * Edx_p2 + (1 - p) * dx**2))
            updates.append((params[i], params[i] + lr*dx))

        return updates
    return train_updates


def sgd(lr=1e-2):
    def train_updates(params, cost):
        gradients = th.grad(cost, params)
        updates = []

        for i in range(len(params)):
            updates.append((params[i], params[i] - lr*gradients[i]))

        return updates

    return train_updates


def momentum(lr=1e-2, p=0.9):
    def train_updates(params, cost):
        gradients = th.grad(cost, params)
        updates = []

        for i in range(len(params)):
            shape = (params[i].get_value(borrow=True)).shape

            old_m = th.shared(np.zeros(shape, dtype='float32'), name='old_m')
            m = p*old_m + (1-p)*gradients[i]

            updates.append((old_m, m))
            updates.append((params[i], params[i] - lr*m))

        return updates

    return train_updates


def CE(decay=0.01):
    def cost(y, out, params):
        weight_penalty = 0
        for param in params:
            weight_penalty += T.sum(param**2)
        return -T.mean(T.log(out[T.arange(y.shape[0]), y.astype('int64')])) + decay*weight_penalty
    return cost


def MSE(decay=0.01):
    def cost(y, out, params):
        weight_penalty = 0
        for param in params:
            weight_penalty += T.sum(param**2)
        return T.mean((y - out)**2) + decay*weight_penalty
    return cost