# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:55:34 2019

@author: Chastine
"""

from ValidationPreprocess import *

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('Matched Features', labeled_img)
    cv2.waitKey()
    cv2.destroyWindow('Matched Features')

#img_bgr = cv2.imread('resources\\Test\\roi.jpg')
img_bgr = cv2.imread('resources\\Templates\\template4.jpg')
img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

pre = ValidationPreprocess()
imgBin = pre.imageToBinary(redefine={'flag':True,'img':img},mode='normal',alg='global')
kernel = np.ones((2,2),np.uint8)
#img_erode = cv2.erode(imgBin,kernel,iterations = 1)
#img_erode = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, kernel)
#imgBin = cv2.Canny(img,100,200)
#cv2.imwrite('coba\\ROInya.jpg',imgBin)

img1, contours, hierarchy = cv2.findContours(imgBin ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for element in contours:
    x,y,w,h = cv2.boundingRect(element)
    if h>w and h>10 and w>5:
        cv2.imwrite(str(h)+'-'+str(w)+'.jpg',img_bgr[y:y+h,x:x+w])
        cv2.rectangle(img_bgr,(x,y),(x+w,y+h),(0,255,0),2)

#ret,labels = cv2.connectedComponents(imgBin)
#print(labels.shape)
#for component in range(ret):
#    print(component)
#    x,y = np.where(labels==component)
#    x,y = list(x),list(y)
#    for i in range(len(x)):
#        labels[x[i]][y[i]]=255
#    cv2.imwrite('coba\\'+str(component)+'.jpg',labels)
##    min_val = min(x)
##    min_val_ind = x.index(min_val)
##    the_point = np.array([x[min_val_ind] , y[min_val_ind]])
##    print("hehe")
##    
##    max_val = max(y)
##    max_val_ind = y.index(max_val)
##    the_pointt = np.array([x[max_val_ind] , y[max_val_ind]])
##    print(the_point,the_pointt)
##    x,y,w,h = cv2.boundingRect(the_point)
#    cv2.rectangle(img_bgr,(the_point[0],the_point[1]),(the_pointt[0],the_pointt[1]),(0,255,0),2)
  
cv2.imshow('Matched Features', img_bgr)
cv2.waitKey()
cv2.destroyWindow('Matched Features')
#imshow_components(labels)
#print(labels)

#cv2.imshow('Matched Features', imgBin)
#cv2.waitKey(0)
#cv2.destroyWindow('Matched Features')