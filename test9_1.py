# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:03:51 2019

@author: fenezema
"""

def pytha(a,b):
    return pow(pow(a,2)+pow(b,2),0.5)

from ValidationPreprocess import *

pojok_kiri_atas = []
pojok_kanan_bawah = []

img = cv2.imread('resources\\Test\\baruu.jpg')
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
pre = ValidationPreprocess()
imgBin = pre.imageToBinary(redefine={'flag':True,'img':imggray})

imggray_row,imggray_col = imgBin.shape
img1, contours, hierarchy = cv2.findContours(imgBin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#ma_x = -1
#ind = None
#for i in range(len(contours)):
#    if len(contours[i])>ma_x:
#        ma_x = len(contours[i])
#        ind = i

#print(len(contours[ind]))
#cv2.drawContours(img,contours, -1, (0,255,0), 3)
for element in contours:
    x,y,w,h = cv2.boundingRect(element)
    pojok_kiri_atas.append([x,y])
    pojok_kanan_bawah.append([w,h])

for i in range(len(pojok_kiri_atas)):
    cv2.rectangle(img,tuple(pojok_kiri_atas[i]),tuple(list(np.asarray(pojok_kiri_atas[i])+np.asarray(pojok_kanan_bawah[i]))),(0,255,0),3)
smaller = cv2.resize(img,(0,0),fx=0.4,fy=0.4)
print(tuple(pojok_kiri_atas[0]),tuple(list(np.asarray(pojok_kiri_atas[0])+np.asarray(pojok_kanan_bawah[0]))))
print(pojok_kiri_atas[0][0],pojok_kanan_bawah[0][0],pojok_kiri_atas[0][1],pojok_kanan_bawah[0][1])
for inds in range(len(pojok_kiri_atas)):
    to_save = img[pojok_kiri_atas[inds][1]:pojok_kiri_atas[inds][1]+pojok_kanan_bawah[inds][1],pojok_kiri_atas[inds][0]:pojok_kiri_atas[inds][0]+pojok_kanan_bawah[inds][0]]
    cv2.imwrite('resources\\Test\\baruu_res'+str(inds)+'.jpg',to_save)
cv2.imshow('IMG', img)
cv2.waitKey(0)
cv2.destroyWindow('IMG')