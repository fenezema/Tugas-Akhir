# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:27:54 2019

@author: Chastine
"""

from core import *

files = os.listdir('resources/PositiveNew_v3/')#os.listdir('resources/NegativeNew/')
#NegativeNew_v3
im = cv2.imread('resources/PositiveNew_v3/'+files[0],0)#cv2.imread('resources/NegativeNew/'+files[0],0)
init = im.shape
flag = 1
for element in files:
    img = cv2.imread('resources/PositiveNew_v3/'+element,0)
    x = img.shape
    if x!=init:
        print(element)
        flag=0
        break

if flag==0:
    print('beda')
else:
    print('sama')