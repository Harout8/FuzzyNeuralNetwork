from builtins import input, str

import cv2

# Для сохранения вышеуказанного файла необходимо импортировать библиотеку «os»
import os

# Создаем экземпляр класса VideoCapture(). Принимает один аргумент - это
# путь к файлу (относительный или абсолютный) или целое число (индекс
# подключенной камеры)
cam = cv2.VideoCapture(0)

cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Загружаем «классификатор
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_alt.xml')

# For each person, enter one numeric face id
face_id = input('\n Введите id пользователя и нажмите Enter ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize individual sampling face count
count = 0

while(True):
    # Функция cap.read() класса VideoCapture() возвращает два объекта:
    # 1) булевое значение (True или False), в случае отсутствия ошибок при
    # загрузке текущего кадра - True. Запишем это в переменную ret
    # 2) сам текущий прочитанный кадр из видео. Запишем его в переменную frame.
    ret, img = cam.read()

    # img = cv2.flip(img, -1) # flip video image vertically

    # Функция cvtColor() конвертирует изображение в нужное цветовое
    # представление. Принимает аргументами сам объект изображения и имя
    # представления, в нашем случае - это черно-белое для уменьшения
    # ресурсозатрат при выводе прочитанного видео на экран.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Классификаторная функция
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # «Маркировать» лица на изображении, используя, например, синий прямоугольник
        # Если грани найдены, они возвращают позиции обнаруженных лиц в виде
        # прямоугольника с левым углом (x, y) и имеют «w» в качестве его ширины
        # и «h» как его высоту ==> (x, y, w, h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        # (для каждого из захваченных кадров мы должны сохранить его как
        # файл в каталоге «dataset»)
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y: y + h, x: x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video

    if k == 27:
        break
    elif count >= 50: # Take 50 face sample and stop video
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")

# Освобождаем оперативную память, занятую переменной cam
cam.release()

# Закрываем все открытые в скрипте окна
cv2.destroyAllWindows()