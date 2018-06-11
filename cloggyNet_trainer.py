from network.cloggyNet import cloggyNet
from cloggy_state_estimator import cloggy_state_estimator
from common import util,functions
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import json

root_path = os.path.dirname(os.path.abspath(__file__))

label_file = open('./data/label.txt', 'rb')
label = pickle.load(label_file)
label_file.close()

label_size = len(label)

training_data_path = os.path.join(os.getcwd(), 'training_data')
testing_data_path = os.path.join(os.getcwd(), 'testing_data')
training_data, training_table = util.setupData(training_data_path, label)
testing_data, testing_table = util.setupData(testing_data_path, label)

training_data = np.array(training_data)
training_table = np.array(training_table)
testing_data = np.array(testing_data)
testing_table = np.array(testing_table)

input_size = training_data.shape[1]
output_size = label_size
hidden_size = round(input_size / 2)


iteration_num = 1000
batch_size = 10
learning_rate = 0.1

train_size = training_data.shape[0]

train_loss_list = []
train_acc_list = []
test_acc_list = []

net = cloggyNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

print("Training start!")
iter_per_epoch = max(round(train_size / batch_size), 1)
for i in range(iteration_num):
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = training_data[batch_mask]
    T_batch = training_table[batch_mask]

    gradient = net.gradient(X_batch, T_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * gradient[key]

    loss = net.loss(X_batch, T_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(training_data, training_table)
        test_acc = net.accuracy(testing_data, testing_table)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("Iteration : ", i + 1)
        print("train acc : ", train_acc)
        print("test acc : ", test_acc)
print("Training end")

print("Saving parameters...")
params_file = open('./data/params.pkl', 'wb')
pickle.dump(net.params, params_file)
params_file.close()
print("Save parameters complete!")
updateDate = datetime.datetime.now()
updateDate = updateDate.__str__()
updateDate = updateDate.split(' ')[0]
result = "success"
json_result = {"updateDate":updateDate, "result":result}
json_path = 'data/result/train/trainedDate.json'
dir_path = os.path.split(root_path)[0]
json_path = os.path.join(dir_path, json_path)
json_file = open(json_path, 'w')
json.dump(json_result, json_file)
json_file.close()