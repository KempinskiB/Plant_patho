import cv2
import numpy as np
import os
from pathlib import Path
#%% attempts
repo_dir = Path(".").absolute()
imgs_dir = repo_dir/"sample_imgs"

img_lst = os.listdir(imgs_dir)

img1_path = os.path.join(imgs_dir, img_lst[2])
img1 = cv2.imread(img1_path,1)

IMG_SIZE = 500
img1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))

edges = cv2.Canny(img1,50,100)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 1)

contours, hierarchy = cv2.findContours(dilation,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
largest = max(contours, key = cv2.contourArea)

color = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img1, largest, -1, (0, 0, 255), 3)

#    cv2.namedWindow("color", cv2.WINDOW_NORMAL)
#    cv2.imshow("original img", color)
cv2.namedWindow("original img", cv2.WINDOW_NORMAL)
cv2.imshow("original img", img1)


x = largest.max(axis = 0)[0][0]
y = largest.max(axis = 0)[0][1]

x = largest.min(axis = 0)[0][0]
y = largest.min(axis = 0)[0][1]

cv2.circle(color,(x,y),10,(0,0,255),-1)
cv2.imshow("dilation", color)

#%%
ret,thresh = cv2.threshold(blur,110,255,0)

#%% "main" sort of
import imutils
def resize_and_multiply_by_rotate(img, ):
    out_folder = repo_dir/"rotated_and_resized"
    IMG_SIZE = 300
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    cv2.imshow('Original image', img)
    for theta in [90,180,270]:
#        out_path=out_folder/str(theta)
        rotated = imutils.rotate(img, 45)

        cv2.imshow(str(theta), rotated)
        if theta == 90:
            flipVertical = cv2.flip(rotated, 0)
            flipHorizontal = cv2.flip(rotated, 1)

    flipVertical = cv2.flip(img, 0)
    flipHorizontal = cv2.flip(img, 1)
    cv2.imshow('vertical image', flipVertical)
    cv2.imshow('horizontal image', flipHorizontal)

#        cv2.imwrite(out_path, rotated)



def find_leaf_contour(img):
    edges = cv2.Canny(img,50,100)
    kernel = np.ones((9,9),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 1)

    contours, hierarchy = cv2.findContours(dilation,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key = cv2.contourArea)

    color = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, largest, -1, (0, 0, 255), 3)

#    cv2.namedWindow("color", cv2.WINDOW_NORMAL)
#    cv2.imshow("original img", color)
    cv2.namedWindow("original img", cv2.WINDOW_NORMAL)
    cv2.imshow("original img", img)

for i in range(1, len(img_lst)):

    img1_path = os.path.join(imgs_dir, img_lst[i])
    img = cv2.imread(img1_path,1)
#    IMG_SIZE = 300
#    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    find_leaf_contour(img)
    if cv2.waitKey(-1) == ord('a'):
        print("a")
        break















