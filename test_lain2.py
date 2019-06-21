# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:08:18 2019

@author: Chastine
"""

from core import *

img = cv2.imread('resources\\Test\\foreground7.jpg')

cv2.rectangle(img,(0,0),(0+100,0+100),(0,255,0),3)

smaller = cv2.resize(img,(0,0),fx=0.4,fy=0.4)

print(smaller.shape)
cv2.imshow('IMG', smaller)
cv2.waitKey(0)
cv2.destroyWindow('IMG')