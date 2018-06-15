from nn.funcs import *
from nn.op import *
import numpy as np

#implements a log loss layer
class loss_layer(op):

    def __init__(self, i_size, o_size):
        super(loss_layer, self).__init__(i_size, o_size)
        self.grads = np.zeros((o_size, i_size))

    def forward(self, x):
        self.x = x
        self.o = softmax(np.dot(x, self.W) + self.b)
        return self.o

    #alpha is used as reward in some reinforcement learning envs
    def backward(self, y, rewards=None):
        one_hot = np.zeros(self.o.shape)
        one_hot[np.arange(self.o.shape[0]), y] = 1
        if rewards is not None:
            self.grads = (one_hot - self.o) * rewards
        else:
            self.grads = one_hot - self.o

    def loss(self, y):
        one_hot = np.zeros(self.o.shape, dtype=np.int)
        one_hot[np.arange(self.o.shape[0]), y] = 1
        #fixed_section = np.nan_to_num((1 - one_hot) * np.log(1 - self.o))
        return -np.mean(np.sum(one_hot * np.log(self.o + 1e-15), axis=1))
