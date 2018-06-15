from nn.funcs import *
from nn.op import *
import numpy as np

class dense(op):

    def __init__(self, i_size, o_size, func_acti, func_acti_grad):
        super(dense, self).__init__(i_size, o_size)
        self.grads = np.zeros((o_size, i_size))
        self.func_acti = func_acti
        self.func_acti_grad = func_acti_grad

    def forward(self, x):
        self.x = x
        self.o = self.func_acti(np.dot(x, self.W) + self.b)
        return self.o

    def backward(self, prev):
        self.grads = self.func_acti_grad(prev.x) * np.dot(prev.grads, prev.W.T)

    def dropout(self, prob):
        self.mask = np.random.binomial(size=self.o.shape[1], n=1, p=1-prob)
        return self.mask
