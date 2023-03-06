import numpy
import cv2
import pickle
from PIL import Image
face_Cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt.xml')
face_Cascade2 = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
paths = {}
with open("labels.pickle", "rb") as f:
   paths = pickle.load(f)
   new_paths = {v:k for k,v in paths.items()} 





wCam, hCam = 1080, 1200
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
Position = False
while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #detecting a face (radius of interest)
    if not Position:
        faces = face_Cascade2.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors = 5)
    #showing face detection 
        for (x, y, w, h,) in faces:
            region_of_interests_grey = gray_img[y:y+h, x:x+w] 

        #recognizing
            path_id, conf = recognizer.predict(region_of_interests_grey)
            photo = Image.open(new_paths[path_id])
            Position = True
            #drawing rectangle
            color = (255, 0, 0)
            stroke = 2
            end_coordinate_x = x + w
            end_coordinate_y = y + h
            cv2.rectangle(img, (x,y),(end_coordinate_x, end_coordinate_y), color)
        
        


    #display video

    if Position:
        picture = numpy.array(photo, "uint8")
        picture = picture[:,:,::-1]
    else:
        picture = img
        
    cv2.imshow('Image', picture)
    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord('r'):
        Position = False 
cap.release()
cv2.destroyAllWindows()