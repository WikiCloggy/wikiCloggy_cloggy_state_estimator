from cloggy_state_estimator import cloggy_state_estimator
from common import util
import pickle
import numpy as np
estimator = cloggy_state_estimator()

label_file = open('./data/label.txt', 'rb')
label = pickle.load(label_file)
label_file.close()

path = './testing_data'

data, table = util.setupData(path, label)

for d in data:
    result = estimator.predict(d)
    index = np.argmax(result)
    print(label[index])

print(estimator.accuracy(data, table))