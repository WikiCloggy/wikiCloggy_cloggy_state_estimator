from cloggy_state_estimator import cloggy_state_estimator
from cloggy_extractor.cloggy_extractor import cloggy_extractor

import pickle
from common import util,functions
import os
import sys
import cv2
import numpy as np
from dogDetector import DogDetector
import json

root_path = os.path.dirname(os.path.abspath(__file__))

def save_result(result, file_path):
    print(result)
    result_file = open(file_path, 'w')
    json.dump(result, result_file)
    result_file.close()

if __name__ == '__main__':
    print("root : " +  root_path)
    flip = None

    if len(sys.argv) < 2:
        print("Need image path")
        exit(0)

    path = sys.argv[1]
    img = cv2.imread(path)
    data_path = os.path.split(root_path)[0]
    data_path = os.path.join(data_path, 'data')

    file_name = os.path.split(path)[1]
    file_name = file_name.split('.')[0]
    print("File name : " + file_name)

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-flip':
            try:
                flip = sys.argv[i + 1]
                if flip == 'True':
                    flip = True
                else:
                    flip = False
            except:
                result = [{'keyword' : 'flip_value_not_found'}]
                result_file_path = os.path.join(data_path, 'result')
                result_file_path = os.path.join(result_file_path, file_name + '.json')
                save_result(result, result_file_path)
                exit(0)

    label_path = os.path.join(root_path, 'data/label.txt')
    label_file = open(label_path, 'rb')
    label = pickle.load(label_file)
    label_file.close()

    data_size = (60, 60)

    extractor = cloggy_extractor()
    img = extractor.optimze_image_size(img)

    estimator = cloggy_state_estimator()

    dog_detector = DogDetector(cfg='tiny-yolo-voc', weights='tiny-yolo-voc', threshold=0.05)
    detect_result = dog_detector.detectsOneDog(img)

    if detect_result == False:
        final_result = [{'keyword' : 'cloggy_not_found'}]

        result_file_path = os.path.join(data_path, 'result')
        result_file_path = os.path.join(result_file_path, file_name + '.json')
        save_result(final_result, result_file_path)

        exit(0)

    input_image_path = os.path.join(data_path, 'input_image')
    input_image_path = os.path.join(input_image_path, file_name + '.jpg')
    cv2.imwrite(input_image_path, img)

    rect = dog_detector.getDogRect(detect_result, img)
    print("Detected dog rect : ", rect)

    if flip is None:
        dog_head_result = dog_detector.detectDogHead(img)
        if not dog_head_result:
            result = [{'keyword' : 'head_not_found'}]
            result_file_path = os.path.join(data_path, 'result')
            result_file_path = os.path.join(result_file_path, file_name + '.json')
            save_result(result, result_file_path)

            exit(0)

        head_rect = dog_detector.getDogPartRect(dog_head_result, img)
        isLeft = dog_detector.isLeft(rect, head_rect)
        if not isLeft:
            flip = True
        else:
            flip = False


    input_silhouette = extractor.delete_background(img, rect)

    input_silhouette = util.resizeImage(input_silhouette, data_size, rect, True)

    if flip:
        input_silhouette = cv2.flip(input_silhouette, 1)

    input_silhouette_path = os.path.join(data_path, 'input_silhouette')
    input_silhouette_path = os.path.join(input_silhouette_path, file_name + '_' + extractor.version + '.png')
    cv2.imwrite(input_silhouette_path, input_silhouette)

    input_silhouette = np.where(input_silhouette > 0, 1, 0)
    input_silhouette = input_silhouette.flatten()

    result = estimator.predict(input_silhouette)
    result = functions.softmax(result[0])
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

    result_file_path = os.path.join(data_path, 'result')
    result_file_path = os.path.join(result_file_path, file_name + '.json')
    save_result(final_result, result_file_path)
