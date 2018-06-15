from nn.funcs import *
import numpy as np

#used as baseline for any differenciable layer
class op():

    def __init__(self, i_size, o_size):
        self.i_size = i_size
        self.o_size = o_size
        self._xavier_init()
        self.o = np.zeros(o_size)
        self.x = None #will be assigned during forward

    def _xavier_init(self):
        self.W = np.random.randn(self.i_size, self.o_size) / np.sqrt(self.i_size)
        self.b = np.random.randn()

    def forward(self, x):
        pass

    def backward(self, prev):
        pass

    def update(self, lr):
        self.W += lr * np.dot(self.x.T, self.grads)
        self.b += lr * np.mean(self.grads, axis = 0)
