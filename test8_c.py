# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:29:35 2019

@author: fenezema
"""

#Impport
from ValidationPreprocess import *
#Import
det_points = {}
count_dp = 0


def segImg(img,nm_fl):
    imge = img
    imggray = cv2.cvtColor(imge,cv2.COLOR_BGR2GRAY)
    pre = ValidationPreprocess()
    imgBin = pre.imageToBinary(redefine={'flag':True,'img':imggray})
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, kernel)
    img1, contours, hierarchy = cv2.findContours(opening ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cou = 0

    for element in contours:
        x,y,w,h = cv2.boundingRect(element)
        if h>w and h>17 and w>4 and h<30:
            cv2.imwrite('coba/charas/'+nm_fl+'-'+str(cou)+'-'+str(h)+'-'+str(w)+'.jpg',img[y:y+h,x:x+w])
            cv2.rectangle(imge,(x,y),(x+w,y+h),(0,255,0),2)
            cou+=1
    return imge,opening

def haarCascade(bgr,flag = False,point=[0,0]):
    detected_points = []
    if flag==True:
        possibleROI = bgr[int(point[1])-50:int(point[1])+50,int(point[0])-125:int(point[0])+125]
    elif flag == False:
        possibleROI = bgr
#    pre = ValidationPreprocess()
#    img = pre.imageToBinary(redefine={'flag':True,'img':img},mode='inverse',alg='global')
    plate_cascade = cv2.CascadeClassifier('D:\\05111540000055_PBaskara\\src\\resources\\CascadeClassifier\\cascadeRGB_v1.xml')
    
    detected = plate_cascade.detectMultiScale(possibleROI,1.03,5)
    for (x,y,w,h) in  detected:
        print(w)
        cv2.rectangle(possibleROI,(x,y),(x+w,y+h),(0,255,0),2)
        detected_points.append([[x,y],[w,h]])
        
#        to_saved = bgrr[y:y+h,x:x+w]
    print(detected_points)
    print('haar finished')
    return possibleROI,detected_points

def doFlannBased(gray,template):
    global det_points,count_dp
    detected_points = []
    sift = cv2.xfeatures2d.SIFT_create()
#    print(template.shape)
    kp_img,des_img = sift.detectAndCompute(gray,None)
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
#            print("Ini yang bagus")
##            print(matches[i])
##            print(m.imgIdx)
##            print(n.imgIdx)
#            print(m.distance)
#            print(kp_template[ m.trainIdx ].pt)
#            print(kp_img[ n.queryIdx ].pt)
            detected_points.append( list(kp_img[ n.queryIdx ].pt) )
#            print('####\n')
            matchesMask[i]=[1,0]
            
#    print('############Selesai##############')       
    draw_params = dict(matchColor = (0,255,0),singlePointColor =  (255,0,0),matchesMask=matchesMask,flags = 0)
    img3 = cv2.drawMatchesKnn(gray,kp_img,template,kp_template,matches,None,**draw_params)
    det_points[str(detected_points)]=len(detected_points)
    print(len(detected_points))
#    print(len(detected_points))
#    print("Drew")

def runInParalel(gray,templates):
    proc = []
    for template in templates:
        p = threading.Thread(target=doFlannBased,args=(gray,template))
        p.start()
        print("Started")

def flannBased(bgr,gray,templates,nm_fl):
    global det_points
    tm_start = time.time()
    checker_flag = 1
    counter = 0
    ind = 0
    
    runInParalel(gray,templates)
#    for template in templates:
#        
#        sift = cv2.xfeatures2d.SIFT_create()
#        print(template.shape)
#        kp_img,des_img = sift.detectAndCompute(gray,None)
#        kp_template,des_template = sift.detectAndCompute(template,None)
#        #kp = sift.detect(gray,None)
#        
#        # FLANN parameters
#        FLANN_INDEX_KDTREE = 1
#        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#        search_params = dict(checks=50)   # or pass empty dictionary
#        
#        flann = cv2.FlannBasedMatcher(index_params,search_params)
#        
#        matches = flann.knnMatch(des_img,des_template,k=2)
#    #    for element in matches:
#    #        print(element)
#        
#        matchesMask = [[0,0] for i in range(len(matches))]
#        #print(matchesMask)
#        
#        for i,(m,n) in enumerate(matches):
#            if m.distance < 0.6*n.distance :#and m.distance/(0.6*n.distance)*100 > 40:
#                print("Ini yang bagus")
#    #            print(matches[i])
#    #            print(m.imgIdx)
#    #            print(n.imgIdx)
#                print(m.distance)
#                print(kp_template[ m.trainIdx ].pt)
#                print(kp_img[ n.queryIdx ].pt)
#                detected_points.append( list(kp_img[ n.queryIdx ].pt) )
#                print('####\n')
#                matchesMask[i]=[1,0]
#                
#        print('############Selesai##############')       
#        draw_params = dict(matchColor = (0,255,0),singlePointColor =  (255,0,0),matchesMask=matchesMask,flags = 0)
#        img3 = cv2.drawMatchesKnn(gray,kp_img,template,kp_template,matches,None,**draw_params)
#        print(len(detected_points))
#        print("Drew")
#        if len(detected_points) <= 3:
#            detected_points = []
#            ind+=1
#            counter+=1
#            if counter==len(templates):
#                checker_flag = 0
#                break
#            continue
#        elif len(detected_points)==4:
#            print("Template : "+str(ind)+'.jpg')
#            afterHaar,pts = haarCascade(bgr,flag=True,point=detected_points[ 1 ])#doHough(img,detected_points[0],bgr)
#            break
#        elif len(detected_points)>2:
#            print("Template : "+str(ind)+'.jpg')
#            afterHaar,pts = haarCascade(bgr,flag=True,point=detected_points[ 4 ])#doHough(img,detected_points[0],bgr)
#            break
##        elif len(detected_points)>1:
##            afterHaar,pts = haarCascade(bgr,detected_points[1])#doHough(img,detected_points[1],bgr)
##            break
#        else:
#            detected_points = []
#            ind+=1
#            counter+=1
#            if counter==len(templates):
#                checker_flag = 0
#                break
#            continue

        
    #cv2.line(img3,(832,204),(1343,204),(0,255,0),5)
    #to_show = cv2.resize(imgHough,(0,0),fx=2.0,fy=2.0)
    #cv2.imwrite('resources\\Test\\sift_img.jpg',img3)
    while len(det_points)<7:
#        print(len(det_points))
        continue
#    detected_points = det_points[max(det_points)]
    print(det_points)
    detected_points = ast.literal_eval(max(det_points.items(), key=operator.itemgetter(1))[0])
    afterHaar,pts = haarCascade(bgr,flag=True,point=detected_points[ 4 ])
    
    ptss = pts[0]
    pka = ptss[0]
    pkb = ptss[1]
    afterHaar1 = afterHaar[pka[1]:pka[1]+pkb[1],pka[0]:pka[0]+pkb[0]]
    cv2.imwrite('coba/charas/roi.jpg',afterHaar1)
    afterHaar2,binlala = segImg(afterHaar1,nm_fl)
    tm_end = time.time()
    print(str(tm_end - tm_start) + ' seconds')
#        if checker_flag == 1:
    cv2.imshow('Matched Features', afterHaar)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')


def main():
    templates = []
    angkanya = 0
    bgrr = cv2.imread('resources\\Test\\foregrounddd'+str(angkanya)+'_frame.jpg')
    grayy = cv2.imread('resources\\Test\\foregrounddd'+str(angkanya)+'_frame.jpg',0)
    for i in range(0,11):
        templates.append(cv2.imread('resources/PositiveNew_v2/proc/imgRef/'+str(i)+'.jpg',0))
    
    row, col, dep = bgrr.shape
    
    bgr = bgrr[int(row/4):int(row/4*3),int(col/3):int(col/3*2)]
    gray = grayy[int(row/4):int(row/4*3),int(col/3):int(col/3*2)]
    #bgrr[int(row/3*2):,:int(col/2)]
    print(bgr.shape)
    print(gray.shape)
    print('====================')
    
    flannBased(bgr,gray,templates,'foregrounddd'+str(angkanya)+'_frame.jpg')
#    after,pts = haarCascade(bgrr)
#    
#    cv2.imshow('Matched Features', after)
#    cv2.waitKey()
#    cv2.destroyWindow('Matched Features')
    
if __name__=="__main__":
    main()