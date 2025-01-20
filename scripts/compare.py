#!/usr/bin/python3.8
import cv2
import time
import os
import tensorflow
from ultralytics import YOLO
from keras.applications import MobileNetV2


mobilenet = tensorflow.keras.models.load_model('/root/bartosz_ws_3/src/receive_image/include/receive_image/model.h5')
yolo = YOLO("/root/bartosz_ws_3/src/receive_image/include/receive_image/yolov8n.pt")
base_mobilenet_model = MobileNetV2(input_shape=(224, 224, 3))

path = '/root/bartosz_ws_3/src/receive_image/include/validation/'

data = []


for image in os.listdir(path):
    image_path = os.path.join(path, image)
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (224, 224))
    data.append(img.reshape((1, 224, 224, 3)))

print("Resizing done")



start_yolo = time.time()
for image in data:
    yolo.predict(image, stream=True, classes=[0], conf=0.5, verbose=False)
stop_yolo = time.time()
print("Yolo time: " + str(stop_yolo-start_yolo))
print("Average single inference time: " + str((stop_yolo-start_yolo)/39))
print("============================")

start = time.time()
for image in data:
    mobilenet.predict(image, verbose=0)
stop = time.time()
print("Mobilenet time: " + str(stop-start))
print("Average single inference time: " + str((stop-start)/39))
print("============================")

start_mobile = time.time()
for image in data:
    base_mobilenet_model.predict(image, verbose=0)
stop_mobile = time.time()

print("Base Mobile Net time: " + str(stop_mobile-start_mobile))
print("Average single inference time: " + str((stop_mobile-start_mobile)/39))
print("============================")

    