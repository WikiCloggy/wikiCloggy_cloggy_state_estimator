from network.cloggyNet import cloggyNet
import pickle
import os

root_path = os.path.dirname(os.path.abspath(__file__))

class cloggy_state_estimator(cloggyNet):
    def __init__(self):
        param_path = os.path.join(root_path, 'data/params.txt')
        file = open(param_path, 'rb')
        self.params = pickle.load(file)
        file.close()