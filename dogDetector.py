from darkflow.net.build import TFNet
from common.singleton import BaseClassSingleton
import dlib

_ERROR_PIXEL = 40

class DogDetector():
    def __init__(self, cfg='yolo', weights='yolov2'):
        self.options = {
            'model': './data/cfg/' + cfg + '.cfg',
            'load': './data/weights/' + weights + '.weights',
            'threshold': 0.3,
            'gpu': 1.0,
            'savepb': True
        }
        print(self.options['model'], self.options['load'])
        self.tfnet = TFNet(self.options)

    def detectsOneDog(self, img):
        result = self.tfnet.return_predict(img)
        for res in result:
            if res['label'] == 'dog':
                return res
        return False

    def detectDogHead(self, img):
        detector = dlib.simple_object_detector("./data/dog_detector.svm")
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
