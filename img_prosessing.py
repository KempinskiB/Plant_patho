import cv2
import numpy as np
import os
from pathlib import Path

imgs_dir = r"C:\Users\lordo\python projects\leaf_clsifier\images"

repo_dir = Path(".").absolute()
imgs_dir = repo_dir/"sample_imgs"

img_lst = os.listdir(imgs_dir)

img1_path = os.path.join(imgs_dir, img_lst[0])
img1 = cv2.imread(img1_path,1)

cv2.namedWindow("img1", cv2.WINDOW_NORMAL) 
cv2.imshow("img1", img1)

edges = cv2.Canny(img1,50,100)
blur = cv2.GaussianBlur(edges,(11,11),0)

ret,thresh = cv2.threshold(blur,110,255,0)

kernel = np.ones((35,35),np.uint8)
dilation = cv2.dilate(thresh,kernel,iterations = 1)
blur = cv2.GaussianBlur(dilation,(19,19),0)

cv2.namedWindow("dilation", cv2.WINDOW_NORMAL) 
cv2.imshow("dilation", blur)

#%%


ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#%%

for i in range(1, len(img_lst)):
    
    img1_path = os.path.join(imgs_dir, img_lst[i])
    img1 = cv2.imread(img1_path,1)
    
    img2_path = os.path.join(imgs_dir, img_lst[i-1])
    img2 = cv2.imread(img2_path,1)
    
#    if compare_imgs(img1, img2) == None:
#        print(i)
#        break
    
    #%%
    

    
    












