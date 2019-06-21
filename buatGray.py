# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:01:04 2019

@author: Chastine
"""

from core import *

dir_ = "resources/temp/"
dir1 = "resources/tempGray_resized/"
files = os.listdir(dir_)
print(files)

for ind in range(len(files)):
    if files[ind]=="proc":
        continue
    img = cv2.imread(dir_+files[ind],0)
    img = cv2.resize(img,(100,50))
    cv2.imwrite(dir1+files[ind],img)