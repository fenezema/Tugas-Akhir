# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 19:05:34 2019

@author: fenezema
"""

from keras.preprocessing.image import ImageDataGenerator
import cv2

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

img = cv2.imread('0.jpg')
img = img.reshape((1,) + img.shape)
i = 0
for batch in datagen.flow(img, batch_size=1, save_to_dir='preview', save_prefix='0', save_format='jpg'):
    i += 1
    if i > 20:
        break