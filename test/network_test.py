import numpy as np
from numpy import testing as npt
from network import SimpleNet as sn, TwoLayerNet as tn
import unittest
import matplotlib.pyplot as plt

class NetworkTest(unittest.TestCase):
    def setUp(self):
        #배열이 오름차순으로 정렬되어있으면 True
        self.T_Label = [True, False]
        self.X_train = np.array([[1, 2, 3, 4, 5],
                                 [3, 6, 7, 8, 20],
                                 [2, 3, 5, 1, 6],
                                 [3, 3, 5, 8 ,1],
                                 [1, 3, 5, 7, 9],
                                 [7, 8, 11, 20, 23],
                                 [3, 2, 1, 4, 5],
                                 [2, 1, 3, 4, 5],
                                 [2, 5, 6, 8, 1],
                                 [3, 4, 5, 6 ,7],
                                 [1, 2, 10, 9, 8],
                                 [1, 2, 4, 3, 5],
                                 [10, 9, 13, 14, 18],
                                 [3, 2, 6, 12, 15],
                                 [3, 5, 6, 7, 5]])
        self.T_Train = np.array([[1, 0],
                                 [1, 0],
                                 [0, 1],
                                 [0, 1],
                                 [1, 0],
                                 [1, 0],
                                 [0, 1],
                                 [0, 1],
                                 [0, 1],
                                 [1, 0],
                                 [0, 1],
                                 [0, 1],
                                 [0, 1],
                                 [0, 1],
                                 [0, 1]])

        self.X_test = np.array([[3, 2, 1, 4, 5],
                                [3, 4, 5, 6, 7],
                                [1, 2, 3, 4, 5]])
        self.T_test = np.array([[0, 1],
                                [1, 0],
                                [1, 0]])

        self.input_size = self.X_train.shape[1]
        self.output_size = self.T_Train.shape[1]

    def testTwoLayerNet(self):
        hidden_size = 4
        net = tn.TwoLayerNet(input_size=self.input_size, output_size=self.output_size, hidden_size=hidden_size)

        self.training(net, iteration_num=2000)

        test_case = [1, 2, 3, 4, 3]
        result = self.predict_bool(net, test_case)
        self.assertEqual(result, False)

    def training(self, network:tn.Network, iteration_num=4000, batch_size=6, learning_rate=0.1):
        train_size = self.X_train.shape[0]

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        iter_per_epoch = max(round(train_size / batch_size), 1)
        print(iter_per_epoch)
        for i in range(iteration_num):
            batch_mask = np.random.choice(train_size, batch_size)
            X_batch = self.X_train[batch_mask]
            T_batch = self.T_Train[batch_mask]

            gradient = network.numerical_gradient(X_batch, T_batch)

            for key in ('W1', 'b1', 'W2', 'b2'):
                network.params[key] -= learning_rate * gradient[key]

            loss = network.loss(X_batch, T_batch)
            train_loss_list.append(loss)

            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(self.X_train, self.T_Train)
                test_acc = network.accuracy(self.X_test, self.T_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("train acc, test acc : " + str(train_acc), str(test_acc))

    def predict_bool(self, network:tn.Network, case):
        result = network.predict(case)
        index = np.argmax(result)
        return self.T_Label[index]

if __name__ == '__main__':
    unittest.main()