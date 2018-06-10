from cloggy_extractor.cloggy_extractor import cloggy_extractor
from common import util
import sys
import os
import cv2
from dogDetector import DogDetector
import time

root_path = os.path.dirname(os.path.abspath(__file__))

def setup_data(command_list):
    img_path = command_list[1]
    img = cv2.imread(img_path)
    flip = None
    keyword = None
    for i in range(2, len(command_list)):
        if command_list[i] == '-flip':
            if command_list[i + 1] == 'True':
                flip = True
            else:
                flip = False
        elif command_list[i] == '-keyword':
            keyword = command_list[i + 1]
    return (img, flip, keyword)

if __name__ == '__main__':
    img, flip, keyword = setup_data(sys.argv)
    if keyword is None:
        print('Keyword is not exist')
        exit(0)

    data_path = 'training_data/' + keyword
    data_path = os.path.join(root_path, data_path)

    extractor = cloggy_extractor()
    detector = DogDetector(cfg='tiny-yolo-voc', weights='tiny-yolo-voc', threshold=0.05)
    time = time.time().__str__()
    time = time.split('.')[0]
    img = extractor.optimze_image_size(img)
    detect_result = detector.detectsOneDog(img)

    if detect_result == False:
        print('cloggy is not found')
        exit(0)
    rect = detector.getDogRect(detect_result, img)

    if flip is None:
        head_detect_result = detector.detectDogHead(img)
        if head_detect_result == False:
            print('Flip is needed')
            exit(0)
        head_rect = detector.getDogPartRect(head_detect_result, img)
        isLeft = detector.isLeft(rect, head_rect)
        if isLeft:
            flip = False
        else:
            flip = True

    silhouette = extractor.delete_background(img, rect)
    silhouette = util.resizeImage(silhouette, (120, 120), rect, True)

    if flip:
        silhouette = cv2.flip(silhouette, 1)

    file_name = keyword + '_' + time
    data_path = os.path.join(data_path, file_name)
    print(data_path)
    cv2.imwrite(data_path, silhouette)