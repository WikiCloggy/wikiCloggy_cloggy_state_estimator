from network.cloggyNet import cloggyNet
import pickle
from common import util
import os
import sys
import cv2
import numpy as np
from dogDetector import DogDetector
import json

from cloggy_extractor.cloggy_extractor import cloggy_extractor

root_path = os.path.dirname(os.path.abspath(__file__))

class cloggy_state_estimator(cloggyNet):
    def __init__(self):
        param_path = os.path.join(root_path, 'data/params.txt')
        file = open(param_path, 'rb')
        self.params = pickle.load(file)
        file.close()

if __name__ == '__main__':
    print("root : " +  root_path)
    path = sys.argv[1]
    img = cv2.imread(path)

    height, width = img.shape[:2]
    if width > 500 or height > 500:
        size = (round(width / 2), round(height / 2))
        img = cv2.resize(img, size, 0, 0, cv2.INTER_LINEAR)

    file_name = os.path.split(path)[1]
    file_name = file_name.split('.')[0]
    print("File name : " + file_name)

    label_path = os.path.join(root_path, 'data/label.txt')
    label_file = open(label_path, 'rb')
    label = pickle.load(label_file)
    label_file.close()

    data_size = (60, 60)

    estimator = cloggy_state_estimator()

    dog_detector = DogDetector(cfg='tiny-yolo-voc', weights='tiny-yolo-voc', threshold=0.08)
    detect_result = dog_detector.detectsOneDog(img)

    if detect_result == False:
        final_result = 'cloggy_not_found'

        result_file_path = 'data/result/' + file_name + '.json'
        result_file_path = os.path.join(root_path, result_file_path)
        result_file = open(result_file_path, 'w')
        json.dump(final_result, result_file)
        result_file.close()

        exit(0)

    rect = dog_detector.getDogRect(detect_result, img)

    print("Detected dog rect : ", rect)
    dog_head_result = dog_detector.detectDogHead(img)
    head_rect = dog_detector.getDogPartRect(dog_head_result, img)
    isDogHeadOnLeft = dog_detector.isLeft(rect, head_rect)

    extractor = cloggy_extractor()
    input_data = extractor.delete_background(img, rect)

    input_data = util.resizeImage(input_data, data_size, rect, True)

    if not isDogHeadOnLeft:
        input_data = cv2.flip(input_data, 1)

    input_data_path = 'data/input/' + file_name + '.png'
    input_data_path = os.path.join(root_path, input_data_path)
    cv2.imwrite(input_data_path, input_data)

    input_data = np.where(input_data > 0, 1, 0)
    input_data = input_data.flatten()

    result = estimator.predict(input_data)
    index = np.argmax(result)

    _label = label.copy()
    final_result = []
    for i in range(3):
        for j in range(i + 1, len(result)):
            if result[i] < result[j]:
                temp = result[i]
                temp_label = _label[i]
                result[i] = result[j]
                result[j] = temp
                _label[i] = _label[j]
                _label[j] = temp_label
        res = {'keyword': _label[i], 'probability': round(result[i], 2)}
        final_result.append(res)

    print(final_result)
    result_file_path = 'data/result/' + file_name + '.json'
    result_file_path = os.path.join(root_path, result_file_path)
    result_file = open(result_file_path, 'w')
    json.dump(final_result, result_file)
    result_file.close()
