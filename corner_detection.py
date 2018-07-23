import cv2 as cv
import numpy as np

# corner detection
# for 3D recreation
# for motion tracking
# character recognition etc

# read the image
img = cv.imread('corner-detection-sample.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray) # turn into float32
# print(gray)
# 100 is how many we're willing to detect
# 0.01 is quality
# 10 is the minimum distance everytime we detect a corner
corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)
# print(corners)

for corner in corners:
    x, y = corner.ravel()
    cv.circle(img, (x, y), 3, 255, -1)
    
cv.imshow('corner', img)