import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_grad(s):
    return s * (1.0 - s)

def relu(x):
    return x * (x > 0)

def  relu_grad(x):
    return 1.0 * (x > 0)

#with numerical stability
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def logloss(x, y):
    probs = softmax(x)
    return probs, -y * np.log(probs)

def logloss_grad(probs, y):
    probs[:,y] -= 1.0
    return probs

def batch_hits(x, y):
    return np.sum(np.argmax(x, axis=1) == y)
