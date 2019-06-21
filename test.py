# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 19:42:49 2019

@author: fenezema
"""
#IMPORT
from ValidationPreprocess import *
from time import sleep
#IMPORT

def haarCascade(img,point,bgr):
    detected_points = []
#    pre = ValidationPreprocess()
#    img = pre.imageToBinary(redefine={'flag':True,'img':img},mode='inverse',alg='global')
    plate_cascade = cv2.CascadeClassifier('D:\\05111540000055_PBaskara\\src\\resources\\CascadeClassifier\\cascadeBinary_v2.xml')
    possibleROI = img[int(point[1])-30:int(point[1])+30,int(point[0])-120:int(point[0])+120]
    bgrr = bgr[int(point[1])-30:int(point[1])+30,int(point[0])-120:int(point[0])+120]
    print(bgrr.shape)
    detected = plate_cascade.detectMultiScale(possibleROI,1.03,5)
    for (x,y,w,h) in  detected:
        cv2.rectangle(possibleROI,(x,y),(x+w,y+h),0,2)
        cv2.rectangle(bgrr,(x,y),(x+w,y+h),(0,255,0),2)
        detected_points.append([[x,y],[x+w,y+h]])
        
        to_saved = bgrr[y:y+h,x:x+w]
#        cv2.imwrite('resources\\Test\\roi.jpg',to_saved)
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
#    cv2.imshow('Matched Features', else_)
#    cv2.waitKey(0)
#    cv2.destroyWindow('Matched Features')

def main():
    indeksnya = 11
    counter = 0
    nextBg = None
    currentBg = None
    foreground = None
    startFlag = 0
    template = cv2.imread('resources\\Templates\\tem_0.jpg',0)
    ret,template = cv2.threshold(template,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    w, h = template.shape[::-1]
    frames = cv2.VideoCapture("D:\\05111540000055_PBaskara\\src\\resources\\Datasets\\zzz-datatest\\test5.mov")
    pre = ValidationPreprocess()
    
    while(True):
        # Capture frame-by-frame
        ret, frame = frames.read()
        
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if startFlag == 0:
            currentBg = gray
            nextBg = cv2.addWeighted(currentBg,0.9,gray,0.1,0)
            startFlag = 1
        else:
            currentBg = nextBg
            nextBg = cv2.addWeighted(currentBg,0.9,gray,0.1,0)#0.9*currentBg)+(0.1*gray)
        
        # Display the resulting frame
        foreground = cv2.subtract(currentBg,gray)
        binFg = pre.imageToBinary(redefine={'flag':True,'img':foreground},mode='normal',alg='otsu',thr=15)
        medianBlurredFg = pre.medianBlur(redefine={'flag':True,'img':binFg})
        morph = pre.morphology(redefine={'flag':True,'img':medianBlurredFg})
        
#        bgrr = frame
#        imgg = binFg
#        fr = gray#foreground1_frame.jpg
#        template = cv2.imread('resources\\\Test\\baruu.jpg',0)
#        row, col = imgg.shape
#        img = imgg[int(row/3*2):,:int(col/2)]
#        fra = fr[int(row/3*2):,:int(col/2)]
#        bgr = bgrr[int(row/3*2):,:int(col/2)]
        
#        flannBased(fra,template,bgr,img)
        
#        res = cv2.matchTemplate(binFg,template,cv2.TM_CCOEFF_NORMED)
#        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#        top_left = max_loc
#        bottom_right = (top_left[0] + w, top_left[1] + h)
#        cv2.rectangle(frame,top_left, bottom_right, (0,255,0), 3)

        #currentBg-gray
        #114-1
        #649-2
        #2923-3
        #2936-4
        #3424-5
        #3491-6
        #3630-7
        #4178-8-taxi
        #7978-9
        #765-10
        #3186-11
        #3797-12
        
        #Motor
        #115
        #164
        #242
        #339
        #474
        #593
        #633
        
        #169
        #725
        #1070
        #1116
        
        #95
        #404
        #823
        #1487
        
        #start ind -> 603
        smaller = cv2.resize(frame,(0,0),fx=0.4,fy=0.4)
        cv2.imshow('frame',smaller)
        counter+=1
#        if counter==115 or counter==164 or counter==242 or counter==339 or counter==474 or counter==593 or counter==633:
#            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+"_frame.jpg",frame)
#            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+"_currentBg.jpg",currentBg)
#            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+".jpg",foreground)
#            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+"_binary.jpg",binFg)
#            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+"_binary_median.jpg",medianBlurredFg)
#            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+"_binary_median_erode_dilate.jpg",morph)
#            print(counter)
#            indeksnya+=1
        if cv2.waitKey(10) & 0xFF == ord('s'):
            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+"_frame.jpg",frame)
            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+"_currentBg.jpg",currentBg)
            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+".jpg",foreground)
            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+"_binary.jpg",binFg)
            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+"_binary_median.jpg",medianBlurredFg)
            cv2.imwrite("resources\\Test\\foregrounddd"+str(indeksnya)+"_binary_median_erode_dilate.jpg",morph)
            print(counter)
            indeksnya+=1
        elif cv2.waitKey(10) & 0xFF == ord('q'):#counter==7978:
#            cv2.imwrite("resources\\Test\\foreground10_frame.jpg",frame)
#            cv2.imwrite("resources\\Test\\foreground10_currentBg.jpg",currentBg)
#            cv2.imwrite("resources\\Test\\foreground10.jpg",foreground)
#            cv2.imwrite("resources\\Test\\foreground10_binary.jpg",binFg)
#            cv2.imwrite("resources\\Test\\foreground10_binary_median.jpg",medianBlurredFg)
#            cv2.imwrite("resources\\Test\\foreground10_binary_median_erode_dilate.jpg",morph)
            break
        
    # When everything done, release the capture
    frames.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()