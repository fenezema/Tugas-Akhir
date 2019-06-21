# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:30:38 2019

@author: Chastine
"""

from ValidationPreprocess import *

pre = ValidationPreprocess()

flag = 1

dir_ = 'D:\\05111540000055_PBaskara\\src\\resources\\Positive\\'

files = os.listdir('resources\\Positive')
temp = cv2.imread('resources\\Positive\\'+files[0],0)
start = temp.shape

#for i in range(8,15):
#    img = cv2.imread(dir_+'pos'+str(i)+'.jpg',0)
#    
#    binImg = cv2.resize(img,(126,30))
#    cv2.imwrite(dir_+'pos'+str(i)+'.jpg',binImg)
#    break
    
for element in files:
    img = cv2.imread('resources\\Positive\\'+element,0)
    if img.shape != start:
        flag=0
        
if flag==1:
    print('tak ada yang berubah')
elif flag==0:
    print('ada yang beda')

     for i in range(8,15):
    img = cv2.imread(dir_+'pos'+str(i)+'.jpg',0)
    
    binImg = pre.imageToBinary(redefine={'flag':True,'img':img},mode='inverse')
    cv2.imwrite(dir_+'pos'+str(i)+'.jpg',binImg)