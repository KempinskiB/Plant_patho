## based on this
## https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from pathlib import Path
import pandas as pd

repo_dir = Path(".").absolute()
sample_imgs_dir = repo_dir/"sample_imgs"
imgs_dir = repo_dir/"all_data"/"images"
map_file = repo_dir/"raw_data"/"train.csv"

map_data = pd.read_csv(map_file)
map_data.set_index('image_id',inplace=True)
map_data = map_data.transpose()

IMG_SIZE = 300

CATEGORIES = ["healthy","multiple_diseases","rust","scab"]

training_data = []

for img in tqdm(os.listdir(imgs_dir)):  # iterate over each image

    # get the img category, and its index
    if "test" in img.lower():
        continue
    img_id = img.strip(".jpg")
    category = map_data[img_id][map_data[img_id]!=0].index[0]
    cat_indx = CATEGORIES.index(category)
    # load the img as an array and resize it
    img_array = cv2.imread(os.path.join(imgs_dir,img))  # convert to array
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    training_data.append([new_array, cat_indx])
    break
    #%%
    cv2.namedWindow("original img", cv2.WINDOW_NORMAL)
    cv2.imshow("original img", new_array)
#%%
## this part can be done pretier
X = []
y = []

for features,label in tqdm(training_data):
    if not label ==1:
        X.append(features)
        y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)
#%%

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#%%

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle
#%%
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
#%
X = X/255.0
#y = y1

#%%
model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",#'categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%
model.fit(X, y, batch_size=5, epochs=3, validation_split=0.1)
#%%# TensorFlow and tf.keras

import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])


model.fit(X, y, epochs=10, validation_split=0.1)


#%%
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
