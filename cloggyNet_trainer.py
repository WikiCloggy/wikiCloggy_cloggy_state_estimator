from network.cloggyNet import cloggyNet
from common import util,functions
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

label_file = open('./data/label.txt', 'rb')
label = pickle.load(label_file)
label_file.close()

training_data = []
training_table = []

label_size = len(label)

for i in range(label_size):
    training_data_path = './training_data/' + label[i]
    for path in glob.glob(os.path.join(training_data_path, '*.png')):
        data = util.loadData(path)
        table = []

        for j in range(label_size):
            if j == i:
                tf = 1
            else:
                tf = 0
            table.append(tf)
        training_data.append(data)
        training_table.append(table)
training_data = np.array(training_data)
training_table = np.array(training_table)

input_size = training_data.shape[1]
output_size = label_size
hidden_size = round(input_size / 2)


iteration_num = 1000
batch_size = 10
learning_rate = 0.1

train_size = training_data.shape[0]

train_loss_list = []
train_acc_list = []
#test_acc_list = []

net = cloggyNet(input_size, hidden_size, output_size)

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
        #test_acc = net.accuracy(testing_data, testing_table)
        train_acc_list.append(train_acc)
        #test_acc_list.append(test_acc)
        print("train acc : " + str(train_acc))
        #print("test acc : " + str(test_acc))
print("Training end")

print("Saving parameters...")
params_file = open('./data/params.txt', 'wb')
pickle.dump(net.params, params_file)
params_file.close()
print("Save parameters complete!")

