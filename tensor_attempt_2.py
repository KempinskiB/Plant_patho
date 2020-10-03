# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, models

import pickle
#%%
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
#%
X = X/255.0
#y = y1
print("done loading")
#%%          FOR CPU RESTRICTION!
#tf.config.threading.set_intra_op_parallelism_threads(2)
#tf.config.threading.set_inter_op_parallelism_threads(2)
#
#with tf.device('/CPU:0'):
#%%

train_images, test_images = X, y

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4))

model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

model.fit(X, y, epochs=10, validation_split=0.1)