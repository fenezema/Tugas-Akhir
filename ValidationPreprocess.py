# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 17:27:31 2019

@author: fenezema
"""

###IMPORT###
from core import *
###IMPORT###


class ValidationPreprocess():
    def __init__(self,filepath=None,filename=None):
        self.filepath = filepath
        self.filename = filename
        
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
    
    def imageToBinary(self,redefine={'flag':False},mode='normal',resizeImg=False,sizeImg=0,alg='otsu',thr=90):
        if redefine['flag']==True:
            try:
                img = redefine['img']
                filepath = '.\\'
                filename = 'temp.jpg'
            except:
                filepath = redefine['filepath']
                filename = redefine['filename']
                img = cv2.imread(filepath+filename,0)
        elif redefine['flag']==False:
            filepath = self.filepath
            filename = self.filename
            img = cv2.imread(filepath+filename,0)
        if alg=='otsu':
            if mode=='inverse':
                ret,bin_image = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            elif mode=='normal':
                ret,bin_image = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif alg=='global':
            if mode=='inverse':
                ret,bin_image = cv2.threshold(img,thr,255,cv2.THRESH_BINARY_INV)
            elif mode=='normal':
                ret,bin_image = cv2.threshold(img,thr,255,cv2.THRESH_BINARY)
        new_image = self.imageResize(bin_image,sizeImg,resizeImg)
        filenm,ext = filename.split('.')
        cv2.imwrite(filepath+filenm+'_new'+'.'+ext,new_image)
        return new_image
        
    def imageToWavelet(self,savedTo_path='.\\'):
        wvType = ['-LL','-LH','-HL','-HH']
        img = cv2.imread(self.filepath+self.filename,0)
        wpImg = pywt.wavedec2(data=img, wavelet='haar', level=2)
        flnm,ext = self.filename.split('.')
        
        cv2.imwrite(savedTo_path+flnm+wvType[0]+'.'+ext,wpImg[0])
        cv2.imwrite(savedTo_path+flnm+wvType[1]+'.'+ext,wpImg[1][0])
        cv2.imwrite(savedTo_path+flnm+wvType[2]+'.'+ext,wpImg[1][1])
        cv2.imwrite(savedTo_path+flnm+wvType[3]+'.'+ext,wpImg[1][2])
        
    def medianBlur(self,redefine={'flag':False},kernel=3):
        if redefine['flag']==True:
            try:
                img = redefine['img']
                filepath = '.\\'
                filename = 'temp.jpg'
            except:
                filepath = redefine['filepath']
                filename = redefine['filename']
                img = cv2.imread(filepath+filename,0)
        elif redefine['flag']==False:
            filepath = self.filepath
            filename = self.filename
            img = cv2.imread(filepath+filename,0)
        img_median = cv2.medianBlur(img,kernel)
        return img_median
    
    def morphology(self,redefine={'flag':False}):
        if redefine['flag']==True:
            try:
                img = redefine['img']
                filepath = '.\\'
                filename = 'temp.jpg'
            except:
                filepath = redefine['filepath']
                filename = redefine['filename']
                img = cv2.imread(filepath+filename,0)
        elif redefine['flag']==False:
            filepath = self.filepath
            filename = self.filename
            img = cv2.imread(filepath+filename,0)
        kernel = np.ones((5,5),np.uint8)
        img_erode = cv2.erode(img,kernel,iterations = 1)
        img_erode_dilate = cv2.dilate(img_erode,kernel,iterations = 1)
        return img_erode_dilate