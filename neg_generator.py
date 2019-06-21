# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:50:01 2019

@author: Chastine
"""

from ValidationPreprocess import *
from random import randint


dir_ = 'D:\\05111540000055_PBaskara\\src\\resources\\Test\\'
dir_save = 'D:\\05111540000055_PBaskara\\src\\resources\\Negative\\'

pre = ValidationPreprocess()

#for row in range(1,13):
#    for col in range(50):
#        img = cv2.imread(dir_+'foreground'+str(row)+'_binary_median.jpg')
#        print(img)
#        x = randint(0,800)
#        to_save = img[x:x+30,x:x+126]
#        cv2.imwrite(dir_save+'foreground'+str(row)+'-'+str(col)+'_binary_median.jpg',to_save)

for i in range(12,35):
    img = cv2.imread('D:\\05111540000055_PBaskara\\opencv-haar-classifier-training\\positive_images\\'+str(i)+'.jpg',0)
    binImg = pre.imageToBinary(redefine={'flag':True,'img':img},mode='inverse')
    cv2.imwrite('D:\\05111540000055_PBaskara\\opencv-haar-classifier-training\\positive_images\\pos_bin\\'+str(i)+'.jpg',binImg)