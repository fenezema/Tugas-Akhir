# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 08:49:48 2019

@author: Chastine
"""
from core import *

def haarCascade(img):
    plate_cascade = cv2.CascadeClassifier('D:\\05111540000055_PBaskara\\src\\resources\\CascadeClassifier\\cascadeBinary.xml')
    print('cascade loaded')
    #possibleROI = img[int(point[1])-30:int(point[1])+30,int(point[0])-120:int(point[0])+120]
    detected = plate_cascade.detectMultiScale(img,1.03,5)
    print('possible ROI detected')
    for (x,y,w,h) in  detected:
        cv2.rectangle(img,(x,y),(x+w,y+h),0,2)
    print('haar finished')
    return img


imgg = cv2.imread('resources\\Test\\foreground2_binary_median.jpg',0)
row,col = imgg.shape
img = imgg[int(row/3*2):,:int(col/2)]

hasil = haarCascade(img)
cv2.imshow('Matched Features', hasil)
cv2.waitKey(0)
cv2.destroyWindow('Matched Features')