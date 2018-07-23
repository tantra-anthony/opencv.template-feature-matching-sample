import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# foreground extractions
img = cv.imread('pug-1.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# in any code, just change this rect value
rect = (161, 79, 150, 150)
# rect = (50, 50, 300, 500)
# take width and heigh * 10% for first 2
# take width and height * 90% for last 2

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()