from cloggy_state_estimator import cloggy_state_estimator
from common import util
import pickle
import numpy as np
estimator = cloggy_state_estimator()

data_path = './testing_data/stomachache/test_skeleton.png'
data = util.loadData(data_path)

result = estimator.predict(data)

label_file = open('./data/label.txt', 'rb')
label = pickle.load(label_file)
label_file.close()

index = np.argmax(result)
print(label[index])