import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

# Face Recognizer, включенный в пакет OpenCV.
recognnizer = cv2.face.LBPHFaceRecognizer_create()

# Загружаем «классификатор
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_alt.xml")

# function to get the images and label data
# (принимать все фотографии в каталоге: «dataset /», возвращая 2 массива:
# «Идентификаторы(lds)» и «Лица(Faces)»)
def getImagesAndLabels(path):
    # os.path.join(path1[, path2[, ...]]) - соединяет пути с учётом
    # особенностей операционной системы.
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')    # convert it to grayscale

        img_numpy = np.array(PIL_img, 'uint8')

        # Метод split разъединяет путь на кортеж, который содержит и файл и каталог.
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Классификаторная функция
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y: y + h, x: x + w])
            ids.append(id)

    return faceSamples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces, ids = getImagesAndLabels(path)

# «Обучить наш распознаватель»
recognnizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognnizer.write('trainer/trainer.yml')    # recognizer.save() worked on Mac, but not on Pi

# Print the number of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))