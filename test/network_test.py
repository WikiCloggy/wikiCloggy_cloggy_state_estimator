import numpy as np
from numpy import testing as npt
from network import SimpleNet as sn, TwoLayerNet as tn
from common import function

#2층 네트워크 테스트
#숫자가 오름차순으로 정렬되 있으면 True를 반환, 아니면 False를 반환
T_Label = [True, False]
X_train = np.array([[1, 2, 3, 4, 5],
                   [3, 6, 7, 8, 20],
                   [2, 3, 5, 1, 6],
                   [3, 3, 5, 8 ,1],
                    [1, 3, 5, 7, 9],
                    [7, 8, 11, 20, 23],
                    [3, 2, 1, 4, 5],
                    [2, 1, 3, 4, 5],
                    [2, 5, 6, 8, 1],
                    [3, 4, 5, 6 ,7]])
T_Train = np.array([[1, 0],
                   [1, 0],
                   [0, 1],
                   [0, 1],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [1, 0]])

X_test = np.array([[3, 2, 1, 4, 5],
                   [3, 4, 5, 6, 7],
                   [1, 2, 3, 4, 5]])
T_test = np.array([[0, 1],
                   [1, 0],
                   [1, 0]])

iters_num = 4000
train_size = X_train.shape[0]
batch_size = 6
learning_rate = 0.1

input_size = X_train.shape[1]
output_size = T_Train.shape[1]

net = tn.TwoLayerNet(input_size=input_size, output_size=output_size, hidden_size=4)

net.params['W1'].shape

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(round(train_size / batch_size), 1)
print(iter_per_epoch)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    T_batch = T_Train[batch_mask]

    gradient = net.numerical_gradient(X_batch, T_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * gradient[key]

    loss = net.loss(X_batch, T_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(X_train, T_Train)
        test_acc = net.accuracy(X_test, T_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc : " + str(train_acc), str(test_acc))

test_case = [15, 10, 18, 20, 22]
result = net.predict(test_case)
idx = np.argmax(result)
result = T_Label[idx]
npt.assert_equal(result, False)