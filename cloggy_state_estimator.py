from network.cloggyNet import cloggyNet
import pickle
import os
from common.layers import *
from collections import OrderedDict

root_path = os.path.dirname(os.path.abspath(__file__))

class cloggy_state_estimator(cloggyNet):
    def __init__(self):
        param_path = os.path.join(root_path, 'data/params.pkl')
        params_file = open(param_path, 'rb')
        self.params = pickle.load(params_file)
        params_file.close()

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()