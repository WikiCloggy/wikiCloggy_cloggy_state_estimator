import numpy as np
import numpy.testing as npt
from common import functions as function
from common import gradient
from common import optimizer

#test sigmoid
X = np.array([-1.0, 1.0, 2.0])
Y = function.sigmoid(X)
expected_Y = np.array([0.26894142, 0.73105858, 0.88079708])
npt.assert_array_almost_equal(Y, expected_Y)

#test ReLU
X = np.array([-0.2, 0.3, 10.0, -2.0])
Y = function.relu(X)
expected_Y = np.array([0, 0.3, 10.0, 0])
npt.assert_array_equal(Y, expected_Y)

#test softmax
a = np.array([0.3, 2.9, 4.0])
Y = function.softmax(a)
expected_Y = np.array([0.01821127, 0.24519181, 0.73659691])
npt.assert_array_almost_equal(Y, expected_Y)

#test MSE
#3번째가 정답
T = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
#3번째일 확률이 가장 높을 것이라고 추정
Y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
MSE = function.MSE(Y, T)
expected_MSE = 0.0975
npt.assert_array_almost_equal(MSE, expected_MSE)

#test CEE
CEE = function.CEE(Y, T)
expected_CEE = 0.510825457099
npt.assert_array_almost_equal(CEE, expected_CEE)

#test numerical_diff
def function_1(x):
    #y = 0.01x^2 + 0.1x
    return 0.01*x**2 + 0.1*x
#0에서 20까지 0.1 간격의 배열 X 생성
X = np.arange(0.0, 20.0, 0.1)
Y = function_1(X)
diff = gradient.numerical_gradient(function_1, 5)
expected_diff = 0.2
npt.assert_almost_equal(diff, expected_diff)

#test numerical_gradient
def function_2(X):
    return np.sum(X**2)
X = np.array([3.0, 4.0])
grad = gradient.numerical_gradient(function_2, X)
expected_gradient = np.array([6, 8])
npt.assert_almost_equal(grad, expected_gradient)

#경사하강법 테스트
init_X = np.array([-3.0, 4.0])
optimized_X = gradient.gradient_descent(function_2, init_X, learning_rate=0.1)
expected_optimized_X = np.array([0, 0])
npt.assert_array_almost_equal(optimized_X, expected_optimized_X)