# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 00:47:05 2021

@author: 13059
"""

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt





labels = ['CT_NonCOVID', 'CT_COVID']
img_size = (224,224)
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path,img))
            resized_arr = cv2.resize(img_arr, (224, 224))
            data.append([resized_arr, class_num])
    return np.array(data)

train = get_data(r"C:\Users\13059\Desktop\2021 Classes\BME6938 MMDM\COVID-CT-master\Images-processed\test")
test = get_data(r"C:\Users\13059\Desktop\2021 Classes\BME6938 MMDM\COVID-CT-master\Images-processed\Images\Test")
val = get_data(r"C:\Users\13059\Desktop\2021 Classes\BME6938 MMDM\COVID-CT-master\Images-processed\Images\Validation")

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)
    
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size = 0.7)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8)

"""
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)
    
for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
"""
# Normalize the data
x_train = np.array(x_train)/255
x_val = np.array(x_val)/255
x_test = np.array(x_test)/255

x_train.reshape(-1,224,224, 1)
y_train = np.array(y_train)

x_val.reshape(-1,224,224, 1)
y_val = np.array(y_val)

x_test.reshape(-1,224,224, 1)
y_test = np.array(y_test)



model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

opt = Adam(lr=0.000001)

model.compile(optimizer = 'adam',
              loss ='binary_crossentropy',
              metrics =['accuracy'])

history = model.fit(x_train, y_train, epochs = 10, verbose = 1, validation_data=(x_val, y_val))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1, -1)[0]
print(classification_report(y_test, predictions, target_names = ['CT_NonCOVID (Class 0)','CT_COVID (Class 1)']))