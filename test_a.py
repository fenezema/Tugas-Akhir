# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:20:39 2019

@author: fenezema
"""
#IMPORT
from ValidationPreprocess import *
from ModelBuild import *
#IMPORT

#GLOBAL INIT
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
model,optimizer = modelBuild()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('saved_weights\\45k_data\\ADAM_0,0001_1000epochs_v3.h5')
labels = {key:chr(key+55) for key in range(10,36)}
#GLOBAL INIT


def getChara(img):
    data_test = []
    pre = ValidationPreprocess()
    imgBin = pre.imageToBinary(redefine={'flag':True,'img':img}, resizeImg = True, sizeImg = 32)
    
    data_test.append(imgBin)
    
    data_test = np.reshape(data_test, (len(data_test), 32, 32, 1))
    
    res = model.predict(data_test)
    pred = return_to_label(res)
    for element in pred:
        if element>9:
            return labels[element]
        else:
            return str(element)
    
def segImg(img,nm_fl):
    the_charas_candidate = {}
    the_charas = []
    imge = img.copy()
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
            the_charas_candidate[x]=[y,[w,h]]
#            cv2.imwrite('coba/charas/'+nm_fl+'-'+str(cou)+'-'+str(h)+'-'+str(w)+'.jpg',img[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    for ind in sorted(the_charas_candidate.keys()):
        temp = the_charas_candidate[ind]
        w,h = temp[1]
        y = temp[0]
        x = ind
#        cv2.imwrite('coba/charas/'+nm_fl+'-'+str(cou)+'-'+str(h)+'-'+str(w)+'.jpg',imge[y:y+h,x:x+w])
        charnya = getChara( imggray[y:y+h,x:x+w] )
        the_charas.append( charnya )
        cou+=1
#    print(the_charas)
    return img,opening,the_charas

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
        cv2.rectangle(possibleROI,(x,y),(x+w,y+h),(0,255,0),2)
        detected_points.append([[x,y],[w,h]])
        
#        to_saved = bgrr[y:y+h,x:x+w]
#        cv2.imwrite('resources\\Test\\roi.jpg',to_saved)
    """print(detected_points)
    print('haar finished')"""
    return possibleROI,detected_points,[int(point[0]-125),int(point[1]-50)]


def flannBased(bgr,gray,templates,nm_fl):
    checker_flag = 1
    counter = 0
    ind = 0
    detected_points = []
    
    for template in templates:
        sift = cv2.xfeatures2d.SIFT_create()
        """print(template.shape)"""
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
                """print("Ini yang bagus")
    #            print(matches[i])
    #            print(m.imgIdx)
    #            print(n.imgIdx)
                print(m.distance)
                print(kp_template[ m.trainIdx ].pt)
                print(kp_img[ n.queryIdx ].pt)"""
                detected_points.append( list(kp_img[ n.queryIdx ].pt) )
                """print('####\n')"""
                matchesMask[i]=[1,0]
                
        """print('############Selesai##############')"""
        draw_params = dict(matchColor = (0,255,0),singlePointColor =  (255,0,0),matchesMask=matchesMask,flags = 0)
        img3 = cv2.drawMatchesKnn(gray,kp_img,template,kp_template,matches,None,**draw_params)
        """print(len(detected_points))
        print("Drew")"""
        if len(detected_points) <= 3:
            detected_points = []
            ind+=1
            counter+=1
            if counter==len(templates):
                checker_flag = 0
                break
            continue
        elif len(detected_points)==4:
            afterHaar,pts, exp_coor = haarCascade(bgr,flag=True,point=detected_points[ 1 ])#doHough(img,detected_points[0],bgr)
            break
        elif len(detected_points)>2:
            afterHaar,pts, exp_coor = haarCascade(bgr,flag=True,point=detected_points[ int(len(detected_points)/2 - 1) ])#doHough(img,detected_points[0],bgr)
            break
#        elif len(detected_points)>1:
#            afterHaar,pts = haarCascade(bgr,detected_points[1])#doHough(img,detected_points[1],bgr)
#            break
        else:
            detected_points = []
            ind+=1
            counter+=1
            if counter==len(templates):
                checker_flag = 0
                break
            continue

        
    #cv2.line(img3,(832,204),(1343,204),(0,255,0),5)
    #to_show = cv2.resize(imgHough,(0,0),fx=2.0,fy=2.0)
    #cv2.imwrite('resources\\Test\\sift_img.jpg',img3)
    
    if checker_flag == 1:
        try:
            ptss = pts[0]
            pka = ptss[0]
            pka1 = [pka[0],pka[1]-5]
            pkb = ptss[1]
            afterHaar1 = afterHaar[pka[1]:pka[1]+pkb[1],pka[0]:pka[0]+pkb[0]]
            afterHaar2,binlala,the_charas = segImg(afterHaar1,nm_fl)
            license_plates_string = ''.join(the_charas)
            print("Detected plates string : " + license_plates_string)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(afterHaar,license_plates_string,tuple(pka1), font, 1,(0,255,0),2,cv2.LINE_AA)

            """print(checker_flag)"""
            print(ptss)
            return {'flag':True,'data':[ pka[0]+exp_coor[0],pka[1]+exp_coor[1],pkb[0],pkb[1] ]}
        except:
           return {'flag':False,'data':"No detected ROI found."}
    elif checker_flag == 0:
        return {'flag':False,'data':"No detected ROI found."}
#        print(checker_flag)
#        #return None,None

def main():
    isTracked = False
    p0 = None
    old_frame_gray = None
    templates = []
#    frames = cv2.VideoCapture("D:\\05111540000055_PBaskara\\src\\resources\\Datasets\\zzz-datatest\\test3.mov")
    fvs = FileVideoStream('D:\\05111540000055_PBaskara\\src\\resources\\Datasets\\zzz-datatest\\test3.mov',queue_size=2048).start()
    pre = ValidationPreprocess()
#    for i in range(0,11):
#        templates.append(cv2.imread('resources/PositiveNew_v2/proc/imgRef/'+str(i)+'.jpg',0))
    templates = [ cv2.imread('resources/PositiveNew_v2/proc/imgRef/NewRef.jpg',0) ]
    counter = 0
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
    
    while fvs.more():
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
        # Capture frame-by-frame
#        ret, frame = frames.read()
        frame = fvs.read()
        
#        # Our operations on the frame come here
        grayy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        row, col, dep = frame.shape
    
        bgr = frame[int(row/4):int(row/4*3),int(col/3):int(col/3*2)]
        gray = grayy[int(row/4):int(row/4*3),int(col/3):int(col/3*2)]
        resss = flannBased(bgr,gray,templates,str(counter))
        counter+=1
        if isTracked == False:
            #afterHaar,binlala = flannBased(bgr,gray,templates,str(counter))
            resss = flannBased(bgr,gray,templates,str(counter))
            if resss['flag']==True:
                isTracked = True
                x_lokasi,y_lokasi,w_real,h_real = resss['data']
                x_real = x_lokasi + int(col/3)
                y_real = y_lokasi +int(row/4)
                p0 = np.array([[[x_real,y_real]],[[x_real + w_real,y_real]],[[x_real,y_real + h_real]],[[x_real + w_real,y_real + h_real]]],dtype="float32")
                old_frame_gray = grayy
            else:
                isTracked = False
#                print(resss['data'])
        elif isTracked == True:
#            print(isTracked)
            frame_gray = grayy
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st==1]
#            print(good_new)
            
            try:
                kiri_pt = (good_new[0][0],good_new[0][1])
                kanan_pt = (good_new[3][0],good_new[3][1])
                cv2.rectangle(frame,kiri_pt,kanan_pt,(255,0,0),2)
            except:
                isTracked = False
            old_frame_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
#        if startFlag == 0:
#            currentBg = gray
#            nextBg = cv2.addWeighted(currentBg,0.9,gray,0.1,0)
#            startFlag = 1
#        else:
#            currentBg = nextBg
#            nextBg = cv2.addWeighted(currentBg,0.9,gray,0.1,0)#0.9*currentBg)+(0.1*gray)
#        
#        # Display the resulting frame
#        foreground = cv2.subtract(currentBg,gray)
#        binFg = pre.imageToBinary(redefine={'flag':True,'img':foreground},mode='normal',alg='otsu',thr=15)
#        medianBlurredFg = pre.medianBlur(redefine={'flag':True,'img':binFg})
#        morph = pre.morphology(redefine={'flag':True,'img':medianBlurredFg})
        
        #start ind -> 603
#        smaller = cv2.resize(frame,(0,0),fx=0.4,fy=0.4)
#        cv2.imshow('frame',smaller)
        
        cv2.imshow('Matched Features', frame)
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
    cv2.destroyAllWindows()
    fvs.stop()

if __name__=="__main__":
    main()