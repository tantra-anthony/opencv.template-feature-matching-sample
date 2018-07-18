import numpy as np
import cv2 as cv

# template matching
img_bgr = cv.imread('template-matching.jpg')
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

template = cv.imread('template.jpg', 0)
w, h = template.shape[::-1]

res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
# higher to make it stricter
threshold = 0.75
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv.rectangle(img_bgr, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

cv.imshow('detected', img_bgr)