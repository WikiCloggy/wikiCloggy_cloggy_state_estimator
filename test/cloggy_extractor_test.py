from cloggy_extractor.cloggy_extractor import cloggy_extractor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cloggy_extractor.imageProcessor as ip

def show_image(img, gray=False):
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.show()

extractor = cloggy_extractor()
marker_size = 8
skip_pixel = 6
fg_threshold = 0.25
bg_threshold = 0.3

img = cv2.imread('image/test5.jpg')
#show_image(img)

rect = (78, 37, 479, 391)
img = extractor.optimze_image_size(img)
_img = extractor.apply_filter(img)
#show_image(img_filtered)
mask = np.zeros(img.shape[:2], np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
cv2.grabCut(_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

show_image(mask, True)

bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)


fg_color_list, bg_color_list = extractor.extract_color_list(img, mask, rect)

mask = extractor.mark_mask(mask, img, rect,
                      fg_color_list=fg_color_list, bg_color_list=bg_color_list,
                      marker_size=marker_size, skip_pixel=skip_pixel,
                      bg_threshold=bg_threshold, fg_threshold=fg_threshold)

show_image(mask * 80, True)

cv2.grabCut(_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

kernal_size = marker_size
if kernal_size % 2 == 0:
    kernal_size += 1
kernal = np.ones((kernal_size, kernal_size), np.uint8)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernal, 2)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernal)
show_image(mask2, True)

data_size = (120, 120)

data = ip.resizeImage(mask2, data_size, rect, True)
show_image(data, True)

cv2.imwrite('auto.png', data)

data2 = '../cloggy_extractor/images/sample_dog2.jpg'
rect2 = (61, 38, 458, 372)
data2 = cv2.imread(data2)
data2 = extractor.delete_background(data2, rect2)
data2 = ip.resizeImage(data2, data_size, rect2, True)
show_image(data2, True)

data3 = '../cloggy_extractor/images/sample_dog3.jpg'
rect3 = (101, 43, 348, 277)
data3 = cv2.imread(data3)
data3 = extractor.delete_background(data3, rect3)
data3 = ip.resizeImage(data3, data_size, rect3, True)
show_image(data3, True)