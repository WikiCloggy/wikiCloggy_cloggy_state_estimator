from cloggy_state_estimator import cloggy_state_estimator
from cloggy_extractor.cloggy_extractor import cloggy_extractor

import pickle
from common import util
import os
import sys
import cv2
import numpy as np
from dogDetector import DogDetector
import json

root_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    print("root : " +  root_path)

    if len(sys.argv) < 2:
        print("Need image path")
        exit(0)

    path = sys.argv[1]
    img = cv2.imread(path)
    data_path = os.path.split(root_path)[0]
    data_path = os.path.join(data_path, 'data')

    height, width = img.shape[:2]
    if width > 640 or height > 640:
        if width > height:
            ratio = 640 / width
            resize_width = 640
            resize_height = round(height * ratio)
        else:
            ratio = 640 / height
            resize_height = 640
            resize_width = round(width * ratio)
        img = util.resizeImage(img, (resize_width, resize_height), (0, 0, width, height))
        #width = round(width / 2)
        #height = round(height / 2)
        #size = (width, height)
        #img = cv2.resize(img, size, 0, 0, cv2.INTER_LINEAR)

    file_name = os.path.split(path)[1]
    file_name = file_name.split('.')[0]
    print("File name : " + file_name)

    label_path = os.path.join(root_path, 'data/label.txt')
    label_file = open(label_path, 'rb')
    label = pickle.load(label_file)
    label_file.close()

    data_size = (60, 60)

    estimator = cloggy_state_estimator()

    dog_detector = DogDetector(cfg='tiny-yolo-voc', weights='tiny-yolo-voc', threshold=0.03)
    detect_result = dog_detector.detectsOneDog(img)

    if detect_result == False:
        final_result = [{'keyword' : 'cloggy_not_found'}]

        result_file_path = os.path.join(data_path, 'result')
        result_file_path = os.path.join(result_file_path, file_name + '.json')
        result_file = open(result_file_path, 'w')
        json.dump(final_result, result_file)
        result_file.close()

        exit(0)

    input_image_path = os.path.join(data_path, 'input_image')
    input_image_path = os.path.join(input_image_path, file_name + '.jpg')
    cv2.imwrite(input_image_path, img)

    rect = dog_detector.getDogRect(detect_result, img)

    print("Detected dog rect : ", rect)
    dog_head_result = dog_detector.detectDogHead(img)
    head_rect = dog_detector.getDogPartRect(dog_head_result, img)
    isDogHeadOnLeft = dog_detector.isLeft(rect, head_rect)

    extractor = cloggy_extractor()
    input_silhouette = extractor.delete_background(img, rect)

    input_silhouette = util.resizeImage(input_silhouette, data_size, rect, True)

    if not isDogHeadOnLeft:
        input_silhouette = cv2.flip(input_silhouette, 1)

    input_silhouette_path = os.path.join(data_path, 'input_silhouette')
    input_silhouette_path = os.path.join(input_silhouette_path, file_name + '.png')
    cv2.imwrite(input_silhouette_path, input_silhouette)

    input_silhouette = np.where(input_silhouette > 0, 1, 0)
    input_silhouette = input_silhouette.flatten()

    result = estimator.predict(input_silhouette)
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
    result_file_path = os.path.join(data_path, 'result')
    result_file_path = os.path.join(result_file_path, file_name + '.json')
    result_file = open(result_file_path, 'w')
    json.dump(final_result, result_file)
    result_file.close()
