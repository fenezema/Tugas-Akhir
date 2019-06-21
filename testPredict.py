# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 12:22:21 2019

@author: fenezema
"""

###IMPORT###
from ValidationPreprocess import *
from ModelBuild import *
###IMPORT###


model,optimizer = modelBuild()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('saved_weights\\45k_data\\ADAM_0,0001_1000epochs_v3.h5')

data_test = []
labels = {key:chr(key+55) for key in range(10,36)}

#inp = input("Nama file : ")
#pre = ValidationPreprocess("resources\\Test\\",inp)
#pre.imageToWavelet(savedTo_path='resources\\Test\\')
while True:
    inp = input("Masukkan nama file : ")
    if inp=='q':
        break
    pre = ValidationPreprocess(filepath="resources\\Test\\",filename=inp)

    newImg = pre.imageToBinary(resizeImg = True,sizeImg=32)
    data_test.append(newImg)
    
    data_test = np.reshape(data_test, (len(data_test), 32, 32, 1))
    
    print(newImg.shape)
    
    res = model.predict(data_test)
    pred = return_to_label(res)
    for element in pred:
        if element>9:
            print(labels[element])
        else:
            print(element)
    data_test=[]