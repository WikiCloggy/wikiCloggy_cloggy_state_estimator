import sys, os
sys.path.append(os.pardir)
import numpy as np
from common import function

class SimpleNet:
    def __init__(self):
        #정규분포로 초기화
        self.W = np.random.randn(2, 3)

    def predict(self, X):
        return np.dot(X, self.W)

    def loss(self, X, T):
        Z = self.predict(X)
        Y = function.softmax(Z)
        loss = function.CEE(Y, T)

        return loss