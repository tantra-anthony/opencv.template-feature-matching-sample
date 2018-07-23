import cv2 as cv
import numpy as np

# haar cascades are massive XML files with massive
# feature types
# can import haar cascades from github (intel license)

# define the cascades
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # the numbers at the back is the parameters for the likelihood of detecting
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # draw rectangle so we get an indicator
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # eye cannot be out of face
        # insert for loop of eye inside the for loop inside the face
        roi_gray = gray[y:y + h, x:x + w]
        # superimpose original image
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # detect eye location
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            
        
    cv.imshow('img', img)
    k = cv.waitKey(30) & 0xFF
    
    if k == 27:
        break
    
cap.release()
cv.destroyAllWindows()