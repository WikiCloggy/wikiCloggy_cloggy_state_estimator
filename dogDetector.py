from darkflow.net.build import TFNet
from common.singleton import BaseClassSingleton
import dlib
import os
import numpy as np



root_path = os.path.dirname(os.path.abspath(__file__))

class DogDetector():
    def __init__(self, cfg='yolo', weights='yolov2', threshold=0.3):
        model_path = os.path.join(root_path, 'cfg/' + cfg + '.cfg')
        load_path = os.path.join(root_path, 'weights/' + weights + '.weights')
        self.options = {
            'model': model_path,
            'load': load_path,
            'threshold': threshold,
            'gpu': 1.0,
            'savepb': True
        }
        print(self.options['model'], self.options['load'])
        self.tfnet = TFNet(self.options)

    def detectsOneDog(self, img):
        result = self.tfnet.return_predict(img)
        dog_list = []
        for res in result:
            if res['label'] == 'dog':
                dog_list.append(res)
        if len(dog_list) == 0:
            return False
        else:
            print(dog_list)
            temp_area = 0
            index = 0
            for i in range(len(dog_list)):
                rect = self.getDogRect(dog_list[i], img)
                area = rect[2] * rect[3]
                if area > temp_area:
                    temp_area = area
                    index = i

            print(dog_list[index])
            return dog_list[index]

    def detectDogHead(self, img):
        detector = dlib.simple_object_detector(os.path.join(root_path, "data/dog_detector.svm"))
        dets = detector(img)

        if(len(dets) == 0):
            print("Can't detect Head")
            dets_pop=dlib.rectangle(0,0,0,0)
        else :
            dets_pop = dets.pop()
        return dets_pop

    def isLeft(self, dog_rect, head_rect):
        if(head_rect[0] + head_rect[2] > dog_rect[0]+dog_rect[2]):
            return False;
        else:
            return True;

    def getDogRect(self, result, originalImg):
        tl = result['topleft']
        br = result['bottomright']


        width = br['x'] - tl['x']
        height = br['y'] - tl['y']
        error_pixel = round(max(width, height) / 8)
        width += error_pixel
        height += error_pixel
        x = max(tl['x'] - round(error_pixel / 2), 1)
        y = max(tl['y'] - round(error_pixel / 2), 1)

        imgHeight, imgWidth = originalImg.shape[:2]

        if (x + width) >= imgWidth:
            width = imgWidth - x - 1
        if (y + height) >= imgHeight:
            height = imgHeight - y - 1
        return (x, y, width, height)

    def getDogPartRect(self, result, originalImg):
        t = result.top()
        l = result.left()
        b = result.bottom()
        r = result.right()

        print ( t, l, b, r)

        width = r - l
        height = b - t
        error_pixel = round(max(width, height) / 8)
        width += error_pixel
        height += error_pixel
        x = max(l - round(error_pixel / 2), 0)
        y = max(t - round(error_pixel / 2), 0)

        imgHeight, imgWidth = originalImg.shape[:2]

        if (x + width) >= imgWidth:
            width = imgWidth - x - 1
        if (y + height) >= imgHeight:
            height = imgHeight - y - 1
        return (x, y, width, height)
