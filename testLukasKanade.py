# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:52:00 2019

@author: fenezema
"""

from core import *

def drawHere(img):
    row,col,ch = img.shape
    possibleROI = img[int(row/3*2):row,:int(col/2)]
    
    cv2.rectangle(possibleROI,(0,0),(30,30),(0,255,0),2)
    
    return possibleROI
    
def main():
    flagger = 1
    cap = cv2.VideoCapture('resources/Datasets/zzz-datatest/test2.mp4')
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = np.array([[[800.0,200.0]],[[1100.0,200.0]],[[800.0,600.0]],[[1100.0,600.0]]],dtype="float32")#cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)#
    print(p0)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while(1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        if flagger == 1:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            print("ini st")
            print(st)
            # Select good points
            good_new = p1[st==1]
#            print("ini gud new")
#            print(good_new)
            good_old = p0[st==1]
            print("ini gud old")
            print(p0[st==1])
        # draw the tracks
#        for i,(new,old) in enumerate(zip(good_new,good_old)):
#            a,b = new.ravel()
#            c,d = old.ravel()
#            try:
#                kiri_pt = (good_new[0][0],good_new[0][1])
#                kanan_pt = (good_new[3][0],good_new[3][1])
#            except:
#                flagger=0
#            #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#            #frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#            cv2.rectangle(frame,kiri_pt,kanan_pt,(0,255,0),2)
        
        try:
            kiri_pt = (good_new[0][0],good_new[0][1])
            kanan_pt = (good_new[3][0],good_new[3][1])
            cv2.rectangle(frame,kiri_pt,kanan_pt,(0,255,0),2)
        except:
#            print(good_new)
            flagger = 0
        
        #img = cv2.add(frame,mask)
        cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    cv2.destroyAllWindows()
    cap.release()
#    img = cv2.imread('baru.jpg')
#    #gray = cv2.cvtColor(imgg,cv2.COLOR_BGR2GRAY)
#    row,col,ch = img.shape
#    imgg = img[int(row/3*2):row,:int(col/2)]
#    
#    newNya = drawHere(imgg)
#
#    cv2.imshow('hehe',imgg)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()