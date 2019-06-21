# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 23:56:38 2019

@author: Chastine
"""

from ValidationPreprocess import *

img1 = cv2.imread('resources\\Test\\foreground2_frame.jpg')
img = cv2.imread('resources\\Test\\foreground2_binary.jpg',0)
img_row,img_col = img.shape
region_row = int(img_row/3*2)
img = img[int(img_row/3*2):,:int(img_col/2)]
#img1 = cv2.imread('resources\\Test\\foreground5_hasil.jpg')
#img = cv2.imread('resources\\Test\\foreground5_hasil.jpg',0)

for ind in range(3):
    template = cv2.imread('resources\\Templates\\template'+str(ind)+'.jpg',0)
    ret,template = cv2.threshold(template,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    w, h = template.shape[::-1]
    
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_val,max_val,min_loc,max_loc)
    if max_val > 0.4:
        top_left = max_loc
        top_left = (top_left[0],top_left[1]+region_row)
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img1,top_left, bottom_right, (0,255,0), 3)
        break

smaller = cv2.resize(img1,(0,0),fx=0.4,fy=0.4)
cv2.imshow('IMG', smaller)
cv2.waitKey(0)
cv2.destroyWindow('IMG')