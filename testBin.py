# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:10:41 2019

@author: fenezema
"""

from ValidationPreprocess import *
from skimage.morphology import skeletonize

img = cv2.imread('resources/PositiveNew_v2/proc/foregrounddd4_frame.jpg',0)

pre = ValidationPreprocess()

imgBin = pre.imageToBinary(redefine={'flag':True,'img':img})

kernel = np.ones((3,3),np.uint8)
kernel1 = np.ones((1,1),np.uint8)
#res = cv2.dilate(imgBin,kernel,iterations = 1)
img_erode = cv2.erode(imgBin,kernel,iterations = 1)
eroded = img_erode.copy()

eroded[eroded == 255] = 1
eroded[eroded == 0] = 0
thinned = np.asarray(skeletonize(eroded),dtype=np.uint8)


thinned[thinned == np.True_] = 255
thinned[thinned == np.False_] = 0

dilated = cv2.dilate(thinned,kernel,iterations = 1)


cv2.imshow('Dilated',dilated)
cv2.imshow('erode',img_erode)
cv2.imshow('ori',img)
#cv2.imshow('Res1',img_erode_dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()