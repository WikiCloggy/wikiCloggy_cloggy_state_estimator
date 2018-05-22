import sys, os
sys.path.append(os.pardir)
from common import function as f
import numpy as np
from network.Network import Network

class TwoLayerNet(Network):
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        #1층 가중치와 편향
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        #2층 가중치와 편향
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, X):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(X, W1) + b1
        z1 = f.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        Y = f.softmax(a2)

        return Y

    def loss(self, X, T):
        Y = self.predict(X)

        return f.CEE(Y, T)

    def accuracy(self, X, T):
        Y = self.predict(X)
        Y = np.argmax(Y, axis=1)
        T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, T):
        loss_W = lambda W: self.loss(X, T)
        gradients = {}
        gradients['W1'] = f.numerical_gradient(loss_W, self.params['W1'])
        gradients['W2'] = f.numerical_gradient(loss_W, self.params['W2'])
        gradients['b1'] = f.numerical_gradient(loss_W, self.params['b1'])
        gradients['b2'] = f.numerical_gradient(loss_W, self.params['b2'])

        return gradients