import cv2
import numpy as np
face_cap = cv2.CascadeClassifier("./venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
myVideo = cv2.VideoCapture(0)
while True:
    success, image = myVideo.read()
    col = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor = 1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for(x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)


    cv2.imshow("Video", image)
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break
myVideo.release()

