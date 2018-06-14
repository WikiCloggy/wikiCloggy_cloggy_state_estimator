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

img = cv2.imread('image/server_test.jpg')
#show_image(img)

rect = (125, 179, 164, 192)

img_filtered = extractor.apply_filter(img)
#show_image(img_filtered)

#around_rect_colors = extractor.extract_color_around_rect(img_filtered, rect)
fg_colors = extractor.extract_foreground_color(img_filtered, rect)
color_map = around_rect_colors.reshape((around_rect_colors.shape[0], 1, 3))
show_image(color_map)

marked_image = img_filtered.copy()
extractor.mark_image(marked_image, marked_image, rect, around_rect_colors, 4, 6, 3)
for y in range(59, 59 + 259, 6):
    for x in range(59, 59 + 365, 6):
        for i in range(fg_colors.shape[0]):
            diff_mean = fg_colors[i] - marked_image[y, x]
            diff_mean = np.mean(diff_mean)
            if diff_mean < 3:
                marked_image = cv2.circle(marked_image, (x, y), 4, (255, 255, 255), -1)
            if abs(diff_mean) > 100:
                marked_image = cv2.circle(marked_image, (x, y), 4, 0, -1)

#extractor.mark_image(marked_image, marked_image, rect, fg_colors, 4, 6, 3, (255, 255, 255))
show_image(marked_image)


img_silhouette = extractor.delete_background(img, rect, marker_size=4, skip_pixel=6, threshold=3)
show_image(img_silhouette, True)

data_size = (120, 120)

data = ip.resizeImage(img_silhouette, data_size, rect, True)
show_image(data, True)

cv2.imwrite('auto.png', data)

data2 = '../cloggy_extractor/images/sample_dog2.jpg'
rect2 = (61, 38, 458, 372)
data2 = cv2.imread(data2)
data2 = extractor.delete_background(data2, rect2, marker_size=4, skip_pixel=6, threshold=3)
data2 = ip.resizeImage(data2, data_size, rect2, True)
show_image(data2, True)

data3 = '../cloggy_extractor/images/sample_dog3.jpg'
rect3 = (101, 43, 348, 277)
data3 = cv2.imread(data3)
data3 = extractor.delete_background(data3, rect3, marker_size=4, skip_pixel=6, threshold=3)
data3 = ip.resizeImage(data3, data_size, rect3, True)
show_image(data3, True)