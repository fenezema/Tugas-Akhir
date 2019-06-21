# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:54:02 2019

@author: Chastine
"""

from ValidationPreprocess import *

#Cek kesamaan ukuran
def makeNeg(files,act_dir,save_dir):
    print(len(files))
    for element in files:
        if element=='proc':
            continue
        else:
            img = cv2.imread(act_dir+element)
            for i in range(42):
                x = randint(200,700)
                imgg = img[x:x+250,x:x+250]
                #binImg = pre.imageToBinary(redefine={'flag':True,'img':imgg},alg='otsu',mode='normal')
                cv2.imwrite(save_dir+str(i)+'-'+element,imgg)

pre = ValidationPreprocess()
act_dir = 'resources/PositiveNew_v2/'
save_dir = 'resources/NegativeNew_v2/'

files = os.listdir(act_dir)

makeNeg(files,act_dir,save_dir)
#makePos()
#Cek kesamaan ukuran

#files = os.listdir('coba/charas')
#
#minn=99999
#maxx=-9999
#
#r_max = None
#r_min = None
#for file in files:
#    img = cv2.imread('coba/charas/'+file,0)
#    row,col = img.shape
#    
#    if row*col > maxx:
#        maxx = row*col
#        r_max = (row,col)
#        
#for file in files:
#    img = cv2.imread('coba/charas/'+file,0)
#    row,col = img.shape
#    
#    if row*col < minn:
#        minn = row*col
#        r_min = (row,col)
#        
#print(r_max,r_min)