import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def ReLU(X):
    return np.maximum(0, X)

def softmax(array):
    # 오버플로우 방지!
    c = np.max(array)
    exp_array = np.exp(array - c)
    sum = np.sum(exp_array)

    return exp_array / sum

#평균 제곱 오차
def MSE(Y, T):
    #Y는 신경망이 추정한 값
    #T는 정답 레이블
    return 0.5 * np.sum((Y - T)**2)

#교차 엔트로피 오차
def CEE(Y, T):
    if Y.ndim == 1:
        T = T.reshape(1, T.size)
        Y = Y.reshape(1, Y.size)

    batch_size = Y.shape[0]
    return -np.sum(T * np.log(Y)) / batch_size
#수치 미분
def numerical_diff(function, x):
    h = 1e-4
    #중심 차분
    return (function(x + h) - function(x - h)) / (2 * h)

def numerical_gradient(function, x):
    h = 1e-4
    gradient = np.zeros_like(x)

    iters = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not iters.finished:
        index = iters.multi_index
        tmp_val = x[index]
        x[index] = float(tmp_val) + h
        fxh1 = function(x)
        x[index] = float(tmp_val) - h
        fxh2 = function(x)
        gradient[index] = (fxh1 - fxh2) / (2 * h)
        x[index] = tmp_val
        iters.iternext()
    return gradient

#경사 하강법
def gradient_descent(function, init_X, learning_rate=0.01, step_num=100):
    #function : 최적화 하려는 함수
    #init_X : 초깃값
    #learning_rate : 학습률, 학습률이 너무 크면 큰 값으로 발산, 너무 작으면 거의 갱신되지 않음
    #step_num : 경사 하강법 반복횟수
    X = init_X

    for i in range(step_num):
        gradient = numerical_gradient(function, X)
        X -= learning_rate * gradient

    return X