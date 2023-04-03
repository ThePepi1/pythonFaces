import os
from PIL import Image
import numpy
import cv2
import pickle
face_Cascade2 = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')
BASE_directory = os.path.dirname(os.path.abspath(__file__))
images_directory = os.path.join(BASE_directory, 'imgs')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
pictures = []
paths = {}
paths_list = []

for root , dirs, files  in os.walk(images_directory):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            print(path)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            pil_img = Image.open(path).convert("L")
            image_array = numpy.array(pil_img, "uint8")
            if not path in paths:
                paths[path] = current_id
                current_id += 1
            
            id_ = paths[path]
            
            faces = face_Cascade2.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 4)
            for (x, y, w, h) in faces:
                region_of_interests = image_array[y:y+h, x:x+w] 
                print(region_of_interests)
                pictures.append(region_of_interests)
                paths_list.append(id_)

with open("labels.pickle", "wb") as f:
    pickle.dump(paths, f)  

print(len(pictures))
while True:
    cv2.imshow('Image', pictures[3])
    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord('r'):
        Position = False 





recognizer.train(pictures, numpy.array(paths_list))
recognizer.save("trainner.yml")