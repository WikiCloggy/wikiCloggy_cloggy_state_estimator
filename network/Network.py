from abc import *

class Network(metaclass= ABCMeta):
    @abstractmethod
    def predict(self, X):
        pass
    @abstractmethod
    def loss(self, X, T):
        pass
    @abstractmethod
    def accuracy(self, X, Y):
        pass
