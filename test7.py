# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:04:12 2019

@author: fenezema
"""

from ValidationPreprocess import *

pre = ValidationPreprocess()
foreground = cv2.imread('resources\\Test\\foreground2_binary.jpg',0)
fg_median = cv2.medianBlur(foreground,5)
cv2.imwrite('resources\\Test\\foreground2_binary_median.jpg',fg_median)
kernel = np.ones((5,5),np.uint8)
fg_median_erode = cv2.erode(fg_median,kernel,iterations = 1)
cv2.imwrite('resources\\Test\\foreground2_binary_median_erode.jpg',fg_median_erode)
fg_median_erode_dilation = cv2.dilate(fg_median_erode,kernel,iterations = 1)
cv2.imwrite('resources\\Test\\foreground2_binary_median_erode_dilation.jpg',fg_median_erode_dilation)