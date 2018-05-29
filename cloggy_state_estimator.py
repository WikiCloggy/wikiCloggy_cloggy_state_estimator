from network.cloggyNet import cloggyNet
import pickle
import os

class cloggy_state_estimator(cloggyNet):
    def __init__(self):
        print(os.getcwd())
        file = open('./data/params.txt', 'rb')
        self.params = pickle.load(file)
        file.close()
