# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 05:48:28 2019

@author: fenezema
"""

###IMPORT###
from ValidationPreprocess import *
###IMPORT###

def doHough(img,point,bgr):
    possibleROI = img[int(point[1])-30:int(point[1])+30,int(point[0])-120:int(point[0])+120]
    bgr = bgr[int(point[1])-30:int(point[1])+30,int(point[0])-120:int(point[0])+120]
    print(bgr.shape)
    pre = ValidationPreprocess()
    edges = pre.imageToBinary(redefine={'flag':True,'img':possibleROI}) #cv2.Canny(possibleROI,50,150,apertureSize = 3)
    
#    lines = cv2.HoughLines(edges,1,np.pi/180,200)
#    for rho,theta in lines[0]:
#        a = np.cos(theta)
#        b = np.sin(theta)
#        x0 = a*rho
#        y0 = b*rho
#        x1 = int(x0 + 1000*(-b))
#        y1 = int(y0 + 1000*(a))
#        x2 = int(x0 - 1000*(-b))
#        y2 = int(y0 - 1000*(a))
#    
#        cv2.line(possibleROI,(x1,y1),(x2,y2),0,2)
    
    minLineLength = 10
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    print(lines)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(possibleROI,(x1,y1),(x2,y2),0,2)
    
    return edges

def forTest(img,point,bgr):
    pre = ValidationPreprocess()
    img = pre.imageToBinary(redefine={'flag':True,'img':img},mode='normal',alg='otsu')
    detected_points=None
    possibleROI = img#[int(point[1])-30:int(point[1])+30,int(point[0])-120:int(point[0])+120]
    bgrr = bgr#[int(point[1])-30:int(point[1])+30,int(point[0])-120:int(point[0])+120]
    print(bgrr.shape)

    return possibleROI,bgr,detected_points

def haarCascade(img,point,bgr):
    detected_points = []
#    pre = ValidationPreprocess()
#    img = pre.imageToBinary(redefine={'flag':True,'img':img},mode='inverse',alg='global')
    plate_cascade = cv2.CascadeClassifier('D:\\05111540000055_PBaskara\\src\\resources\\CascadeClassifier\\cascadeBinary190x50_v2.xml')
    possibleROI = img[int(point[1])-30:int(point[1])+30,int(point[0])-120:int(point[0])+120]
    bgrr = bgr[int(point[1])-30:int(point[1])+30,int(point[0])-120:int(point[0])+120]
    print(bgrr.shape)
    detected = plate_cascade.detectMultiScale(possibleROI,1.03,5)
    for (x,y,w,h) in  detected:
        cv2.rectangle(possibleROI,(x,y),(x+w,y+h),0,2)
        cv2.rectangle(bgrr,(x,y),(x+w,y+h),(0,255,0),2)
        detected_points.append([[x,y],[x+w,y+h]])
        
        to_saved = bgrr[y:y+h,x:x+w]
        cv2.imwrite('resources\\Test\\roi.jpg',to_saved)
    print('haar finished')
    return possibleROI,bgr,detected_points

def flannBased(img,template,bgr,binary_img):
    detected_points = []
    sift = cv2.xfeatures2d.SIFT_create()
    kp_img,des_img = sift.detectAndCompute(img,None)
    kp_template,des_template = sift.detectAndCompute(template,None)
    #kp = sift.detect(gray,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches = flann.knnMatch(des_img,des_template,k=2)
#    for element in matches:
#        print(element)
    
    matchesMask = [[0,0] for i in range(len(matches))]
    #print(matchesMask)
    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.6*n.distance :#and m.distance/(0.6*n.distance)*100 > 40:
            print("Ini yang bagus")
            print(m.distance,0.6*n.distance,m.distance/(0.6*n.distance)*100)
#            print(matches[i])
#            print(m.imgIdx)
#            print(n.imgIdx)
            print(kp_template[ m.trainIdx ].pt)
            print(kp_img[ n.queryIdx ].pt)
            detected_points.append( list(kp_img[ n.queryIdx ].pt) )
            print('####\n')
            matchesMask[i]=[1,0]
            
    print('############Selesai##############')       
    draw_params = dict(matchColor = (0,255,0),singlePointColor =  (255,0,0),matchesMask=matchesMask,flags = 0)
    img3 = cv2.drawMatchesKnn(img,kp_img,template,kp_template,matches,None,**draw_params)
    if len(detected_points)==1:
        imgHough,else_,pts = haarCascade(binary_img,detected_points[0],bgr)#doHough(img,detected_points[0],bgr)
    elif len(detected_points)>1:
        imgHough,else_,pts = haarCascade(binary_img,detected_points[1],bgr)#doHough(img,detected_points[1],bgr)
    else:
        imgHough,else_,pts = img,img,None

        
    #cv2.line(img3,(832,204),(1343,204),(0,255,0),5)
    #to_show = cv2.resize(imgHough,(0,0),fx=2.0,fy=2.0)
    #cv2.imwrite('resources\\Test\\sift_img.jpg',img3)
    cv2.imshow('Matched Features', else_)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')


angkanya = 13
bgrr = cv2.imread('resources\\Test\\foregrounddd'+str(angkanya)+'_frame.jpg')
imgg = cv2.imread('resources\\Test\\foregrounddd'+str(angkanya)+'_binary.jpg',0)
fr = cv2.imread('resources\\Test\\foregrounddd'+str(angkanya)+'_frame.jpg',0)#foreground1_frame.jpg
template = cv2.imread('resources\\\Templates\\template7.jpg',0)
row, col = imgg.shape
img = imgg[int(row/3):int(row/3*2),int(col/3):int(col/3*2)]#imgg[int(row/3*2):,:int(col/2)]
fra = fr[int(row/3):int(row/3*2),int(col/3):int(col/3*2)]#fr[int(row/3*2):,:int(col/2)]
bgr = bgrr[int(row/3):int(row/3*2),int(col/3):int(col/3*2)]#bgrr[int(row/3*2):,:int(col/2)]
print(bgr.shape)
print(row/3*2,col/2)
print(row,col)
print('====================')

flannBased(fra,template,bgr,img)
#newImg,grayed,pts = haarCascade(img,None,bgr)
#cv2.imshow('Matched Features', grayed)
#cv2.waitKey(0)
#cv2.destroyWindow('Matched Features')
#img=cv2.drawKeypoints(gray,kp,None)
#cv2.imwrite('resources\\Test\\sift_kp.jpg',img)