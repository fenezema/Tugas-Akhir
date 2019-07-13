# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:38:33 2019

@author: fenezema
"""
###IMPORT###
from core import *
###IMPORT###

class ImagePreprocessing:
    def __init__(self,init_working_dir,proc_dir):
        self.directory_path = init_working_dir
        self.directory_path_processed = proc_dir
        self.current_path = None
        self.current_filename = None
        self.current_filename_path = None
    
    def setCurrentPath(self,path_name):
        self.current_path = self.directory_path+path_name+"\\"
        
    def setFilename(self,filename):
        self.current_filename = filename
        self.current_filename_path = self.current_path+filename
    
    def getDirPath(self):
        return self.directory_path
    
    def getCurrentPath(self):
        return self.current_path
    
    def getFilename(self):
        return self.current_filename
    
    def getFilenamePath(self):
        return self.current_filename_path
    
    def imageResize(self,img,size,flag):
        if flag==True:
            hei,wid = img.shape
    
            if hei>wid:
                widPadSize = hei-wid
                leftWidPadSize = int((hei-wid)/2)
                rightWidPadSize = widPadSize-leftWidPadSize
                newimg=cv2.copyMakeBorder(img, top=0, bottom=0, left=leftWidPadSize, right=rightWidPadSize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
            elif wid>hei:
                widPadSize = wid-hei
                topWidPadSize = int((wid-hei)/2)
                bottomWidPadSize = widPadSize-topWidPadSize
                newimg=cv2.copyMakeBorder(img, top=topWidPadSize, bottom=bottomWidPadSize, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
            else:
                newimg = img        
            newly = cv2.resize(newimg,(size,size))
            return newly
        elif flag==False:
            return img
    
    def augmentImages(self,path,flag):
        if flag==True:
            self.datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=False, fill_mode='nearest')
            file_list = os.listdir(path)
            
            for element in file_list:
                prefix,ext = element.split(".")
                self.data = cv2.imread(path+element)
                self.data = self.data.reshape((1,) + self.data.shape)
                i = 0
                for batch in self.datagen.flow(self.data, batch_size=1, save_to_dir=path, save_prefix=prefix, save_format='jpg'):
                    i += 1
                    if i > 20:
                        break
        elif flag==False:
            print("Data Augment has been skipped")
    
    def toBinary(self,resizeImg=False,sizeImg=0):
        self.img = cv2.imread(self.current_filename_path,0)
        print(self.current_filename_path)
        self.ret,self.th = cv2.threshold(self.img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.th = self.imageResize(self.th,sizeImg,resizeImg)
        cv2.imwrite(self.getFilename(),self.th)
        print("Current File : "+self.getFilename())