# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:50:28 2019

@author: fenezema
"""

import cv2

img1 = cv2.imread('resources\\Datasets\\0\\0.jpg',0)
img = cv2.imread('resources\\Datasets\\0\\0_0_515.jpg',0)

hei,wid = img.shape

if hei>wid:
    widPadSize = hei-wid
    leftWidPadSize = int((hei-wid)/2)
    rightWidPadSize = widPadSize-leftWidPadSize
    newimg=cv2.copyMakeBorder(img, top=0, bottom=0, left=leftWidPadSize, right=rightWidPadSize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
elif wid>hei:
    widPadSize = wid-hei
    topWidPadSize = int((wid-hei)/2)
    bottomWidPadSize = widPadSize-topWidPadSize
    newimg=cv2.copyMakeBorder(img, top=topWidPadSize, bottom=bottomWidPadSize, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
else:
    newimg = img        
newly = cv2.resize(newimg,(50,50))
cv2.imshow('new',newly)
cv2.waitKey(0)
cv2.destroyAllWindows()
#newimg = cv2.resize(img,)