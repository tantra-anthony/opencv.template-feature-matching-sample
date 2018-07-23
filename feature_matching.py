import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('feature-matching-template.jpg', 0)
img2 = cv.imread('feature-matching-image.jpg', 0) # can change to 1 for color but mind the RGB and BGR

# define the detector
orb = cv.ORB_create()

# kp = keypoints, des = descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)
#sorting from least to most matches
matches = sorted(matches, key = lambda x:x.distance)

# keep the matches low so we can be most accurate
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags = 2)
plt.imshow(img3)
plt.show()