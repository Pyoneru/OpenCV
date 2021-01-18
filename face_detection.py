import cv2
import numpy as np
import datetime as dt

face_cascade = cv2.CascadeClassifier('.//haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

# Loop end when user press 'ESC'
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        #                                       Red Line
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

        face_gray = frame_gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(face_color, (int(ex+(ew/2)), int(ey+(ew/2))), 20, (0, 255, 0), 2)
    cv2.imshow('Detect face', frame)

    key = cv2.waitKey(30) & 0xff
    if key == 115 or key == 83:
        filename = str(dt.datetime.now().strftime("%Y%m%d%H%M%S")) + '.jpg'
        print('save: ' + filename)
        x = cv2.imwrite(filename, frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()