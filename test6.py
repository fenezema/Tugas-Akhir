# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:37:03 2019

@author: fenezema
"""

from ValidationPreprocess import *

file_ = 'D:\\05111540000055_PBaskara\\src\\resources\\Test\\baru.jpg'

preimg = ValidationPreprocess('D:\\05111540000055_PBaskara\\src\\resources\\Test\\','baru.jpg')

binimg = preimg.imageToBinary()
cv2.imwrite('resources\\Test\\newbinimg.jpg',binimg)