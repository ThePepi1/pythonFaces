import os
from PIL import Image
import numpy
import cv2
face_Cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
face_Cascade2 = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')
BASE_directory = os.path.dirname(os.path.abspath(__file__))
images_directory = os.path.join(BASE_directory, 'imgs')

current_id = 0
pictures = []
labels = {}
labels_list = []

for root , dirs, files  in os.walk(images_directory):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            pil_img = Image.open(path).convert("L")
            image_array = numpy.array(pil_img, "uint8")
            if not label in labels:
                labels[label] = current_id
                current_id += 1
            
            id_ = labels[label]
            faces = face_Cascade2.detectMultiScale(image_array, scaleFactor = 1.3, minNeighbors = 5)
            print(faces)
            for (x, y, w, h) in faces:
                region_of_interests = image_array[y:y+h, x:x+w] 
                pictures.append(region_of_interests)
                labels_list.append(id_)



print(labels_list)
print(pictures)