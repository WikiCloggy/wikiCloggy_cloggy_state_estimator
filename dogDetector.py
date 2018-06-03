from darkflow.net.build import TFNet
from common.singleton import BaseClassSingleton
import dlib
import os

_ERROR_PIXEL = 40

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
            return dog_list

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

        x = max(tl['x'] - round(_ERROR_PIXEL / 2), 0)
        y = max(tl['y'] - round(_ERROR_PIXEL / 2), 0)
        width = br['x'] - tl['x'] + _ERROR_PIXEL
        height = br['y'] - tl['y'] + _ERROR_PIXEL

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

        x = max(l - round(_ERROR_PIXEL / 2), 0)
        y = max(t - round(_ERROR_PIXEL / 2), 0)
        width = r - l + _ERROR_PIXEL
        height = b - t + _ERROR_PIXEL

        imgHeight, imgWidth = originalImg.shape[:2]

        if (x + width) >= imgWidth:
            width = imgWidth - x - 1
        if (y + height) >= imgHeight:
            height = imgHeight - y - 1
        return (x, y, width, height)
