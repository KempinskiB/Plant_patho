## based on this
## https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from pathlib import Path
import pandas as pd

DATADIR = "X:/Datasets/PetImages"

repo_dir = Path(".").absolute()
imgs_dir = repo_dir/"sample_imgs"
map_file = repo_dir/"raw_data"/"train.csv"

map_data = pd.read_csv(map_file)
map_data.set_index('image_id',inplace=True)
map_data = map_data.transpose()

IMG_SIZE = 500

CATEGORIES = ["healthy","multiple_diseases","rust","scab"]

training_data = []

for img in tqdm(os.listdir(imgs_dir)):  # iterate over each image

    # get the img category, and its index
    img_id = img.strip(".jpg")
    category = map_data[img_id][map_data[img_id]!=0].index[0]
    cat_indx = CATEGORIES.index(category)
    # load the img as an array and resize it
    img_array = cv2.imread(os.path.join(imgs_dir,img))  # convert to array
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    training_data.append([new_array, cat_indx])

## this part can be done pretier
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)



