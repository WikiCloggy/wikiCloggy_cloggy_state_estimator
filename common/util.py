# coding: utf-8
import numpy as np
import cv2
import glob
import os

def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용
    
    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """데이터셋을 뒤섞는다.

    Parameters
    ----------
    x : 훈련 데이터
    t : 정답 레이블
    
    Returns
    -------
    x, t : 뒤섞은 훈련 데이터와 정답 레이블
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def resizeImage(img, wanted_size, rect=None , maintain_ratio=False):
    if rect is None:
        x, y, width, height = (0, 0, img.shape[1], img.shape[0])
    else:
        x, y, width, height = rect
    data_width, data_height = wanted_size
    img = img[y:y + height, x:x + width]
    shape = list(img.shape)
    shape[0] = data_height
    shape[1] = data_width
    shape = tuple(shape)
    result = np.zeros(shape, dtype=img.dtype)

    if maintain_ratio:
        height_ratio = data_height / height
        resized_width = round(width * height_ratio)
        if resized_width < data_width:
            resized_img_size = (resized_width, data_height)
            img = cv2.resize(img, resized_img_size, 0, 0, cv2.INTER_LINEAR)
            width_space = round((data_width - resized_width) / 2)
            result[:data_height, width_space:width_space + resized_width] = img[:data_height, :resized_width]
        else:
            width_ratio = data_width / width
            resized_height = round(height * width_ratio)
            resized_img_size = (data_width, resized_height)
            img = cv2.resize(img, resized_img_size, 0, 0, cv2.INTER_LINEAR)
            height_space = round((data_height - resized_height) / 2)
            result[height_space:height_space + resized_height, :data_width] = img[:data_height, :data_width]
    else:
        result = cv2.resize(img, wanted_size, 0, 0, cv2.INTER_LINEAR)
    return result

def loadData(path):
    data = cv2.imread(path, 0)
    shape = data.shape
    data = resizeImage(data, (60, 60), maintain_ratio=True)
    data = np.where(data > 0, 1, 0)
    #plt.imshow(data)
    #plt.show()
    data = data.flatten()
    return data

def setupData(data_path, label):
    label_size = len(label)

    data_array = []
    table_array = []

    for i in range(label_size):
        path = os.path.join(data_path, label[i])
        for dt_path in glob.glob(os.path.join(path, '*png')):
            data = loadData(dt_path)
            table = []

            for j in range(label_size):
                if j == i:
                    value = 1
                else:
                    value = 0
                table.append(value)
            data_array.append(data)
            table_array.append(table)

    return (np.array(data_array), np.array(table_array))