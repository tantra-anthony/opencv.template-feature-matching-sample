import numpy as np
import cv2 as cv

# foreground detection
# find places that don't change
# finding motion in images and videos
# finding differences in images and videos

cap = cv.VideoCapture(0)

fgbg = cv.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    cv.imshow('original', frame)
    cv.imshow('fg', fgmask)
    
    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break
    
cap.release()
cv.destroyAllWindows()
