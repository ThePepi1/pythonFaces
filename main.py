import numpy as np
import cv2
face_Cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt.xml')
face_Cascade2 = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

wCam, hCam = 1080, 1200
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #detecting a face (radius of interest)
    faces = face_Cascade2.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 5)
    #showing face detection 
    for (x, y, w, h,) in faces:
        region_of_interests_grey = gray_img[y:y+h, x:x+w] 

        #recognizing
        path_id, conf = recognizer.predict(region_of_interests_grey)
        




        #drawing rectangle
        color = (255, 0, 0)
        stroke = 2
        end_coordinate_x = x + w
        end_coordinate_y = y + h
        cv2.rectangle(img, (x,y),(end_coordinate_x, end_coordinate_y), color)


    #display video
    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()