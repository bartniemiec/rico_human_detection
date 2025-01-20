#!/usr/bin/python3.8
import os
import cv2
import random
import kagglehub
import numpy as np
import pandas as pd
from keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import tensorflow


def read_and_preprocess_image(image_path):
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (224, 224))  # Resize to the input size of MobileNetV2
    return image


def main():

    path = "/root/bartosz_ws_3/src/receive_image/include/dataset"

    data = []
    labels = []


    for class_name in os.listdir(path):
        print("Searching " + str(class_name))
        class_dir = os.path.join(path, class_name)
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            image = read_and_preprocess_image(image_path)
            data.append(image)
            labels.append(int(class_name))

    labels = to_categorical(labels, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    model = Sequential()
    base_model = MobileNetV2(input_shape=(224, 224, 3))
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    batch_size = 16
    epochs = 10

    history = model.fit(datagen.flow(np.array(X_train), y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size,
                        epochs=epochs,
                        validation_data=(np.array(X_test), y_test))

    model.save('/root/bartosz_ws_3/src/receive_image/include/receive_image/model.h5')

if __name__ == "__main__":
    main()