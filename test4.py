# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:25:06 2019

@author: fenezema
"""

import cv2

img = cv2.imread("F:\\Kuliah\\Tugas Akhir\\05111540000055_PBaskara\\src\\resources\\Datasets\\0\\0.jpg")
img1 = cv2.imread("F:\\Kuliah\\Tugas Akhir\\05111540000055_PBaskara\\src\\resources\\Processed\\0\\0.jpg")

print(img.shape,img1.shape)

cv2.imshow('new',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()