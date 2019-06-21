# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 01:37:36 2019

@author: fenezema
"""

def pytha(a,b):
    return pow(pow(a,2)+pow(b,2),0.5)

from core import *

pojok_kiri_atas = []
pojok_kanan_bawah = []

img = cv2.imread('resources\\Test\\foregrounddd6_binary.jpg')
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imggray_row,imggray_col = imggray.shape
img_proc = img[int(imggray_row/3):int(imggray_row/3*2),int(imggray_col/3):int(imggray_col/3*2)]
imggray_proc = imggray[int(imggray_row/3):int(imggray_row/3*2),int(imggray_col/3):int(imggray_col/3*2)]
search_region_row_index = imggray_row*2/3
search_region_col_index = imggray_col/2
imggray_diagonal = pytha(imggray_row,imggray_col)
th_upper = 0.2*imggray_diagonal
th_bottom = 0.1*imggray_diagonal
img1, contours, hierarchy = cv2.findContours(imggray_proc,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
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
#    if y > search_region_row_index and y < imggray_row and x < search_region_col_index:
#        if w>h:
#            temp = pytha(w,h)
#            if temp >= th_bottom and temp < th_upper:
#                    pojok_kiri_atas.append([x,y])
#                    pojok_kanan_bawah.append([w,h])
    if w>h:
        pojok_kiri_atas.append([x,y])
        pojok_kanan_bawah.append([w,h])

for i in range(len(pojok_kiri_atas)):
    cv2.rectangle(img_proc,tuple(pojok_kiri_atas[i]),tuple(list(np.asarray(pojok_kiri_atas[i])+np.asarray(pojok_kanan_bawah[i]))),(0,255,0),3)
#smaller = cv2.resize(img,(0,0),fx=0.4,fy=0.4)
print(tuple(pojok_kiri_atas[0]),tuple(list(np.asarray(pojok_kiri_atas[0])+np.asarray(pojok_kanan_bawah[0]))))
print(pojok_kiri_atas[0][0],pojok_kanan_bawah[0][0],pojok_kiri_atas[0][1],pojok_kanan_bawah[0][1])
inds=0
to_save = img[pojok_kiri_atas[inds][1]:pojok_kiri_atas[inds][1]+pojok_kanan_bawah[inds][1],pojok_kiri_atas[inds][0]:pojok_kiri_atas[inds][0]+pojok_kanan_bawah[inds][0]]
#cv2.imwrite('resources\\Test\\foreground6_hasil.jpg',to_save)
cv2.imshow('IMG', img_proc)
cv2.waitKey(0)
cv2.destroyWindow('IMG')