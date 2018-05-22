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
    #Y는 신경망이 추정한 값
    #T는 정답 레이블
    #-inf 방지
    delta = 1e-7
    return -np.sum(T * np.log(Y + delta))

#수치 미분
def numerical_diff(f, x):
    h = 1e-4
    #중심 차분
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x):
    h = 1e-4
    gradient = np.zeros_like(x)

    for i in range(x.size):
        temp = x[i]

        # f(x + h)
        x[i] = temp + h
        fxh1 = f(x)

        #f(x - h)
        x[i] = temp - h
        fxh2 = f(x)

        gradient[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = temp

    return gradient