#IMPORT
from ValidationPreprocess import *
from ModelBuild import *
#IMPORT

#GLOBAL INIT
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

model,optimizer = modelBuild()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('saved_weights/Kernel_Scenario/2-Kernel3_4x4/Adam_Kernel3_4x4_0,0001_200epochs.h5')
labels = {key:chr(key+55) for key in range(10,36)}
network_path = '\\\\10.151.33.41\\Users\\Chastine\\anotherResources\\'
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
    try:
        print('masuk try')
        imge = img.copy()
    except:
        return 0,0,0,False
    imggray = cv2.cvtColor(imge,cv2.COLOR_BGR2GRAY)
    h_imggray,w_imggray = imggray.shape
    pre = ValidationPreprocess()
    imgBin = pre.imageToBinary(redefine={'flag':True,'img':imggray})

    kernel = np.ones((3,3),np.uint8)
    erode = cv2.erode(imgBin,kernel,iterations = 1)
    dilate = cv2.dilate(erode,kernel,iterations = 2)
    erode1 = cv2.erode(dilate,kernel,iterations = 1)

    # erode = cv2.erode(imgBin,kernel,iterations = 1)
    # dilate = cv2.dilate(erode,kernel,iterations = 3)
    # erode1 = cv2.erode(dilate,kernel,iterations = 2)
    # dilate1 = cv2.dilate(erode1,kernel,iterations = 1)
    # erode2 = cv2.erode(dilate1,kernel,iterations = 1)
    img1, contours, hierarchy = cv2.findContours(erode1 ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cou = 0

    for element in contours:
        x,y,w,h = cv2.boundingRect(element)
        if h>w and w/w_imggray > 0.04 and w/w_imggray <=0.15 and h/h_imggray >= 0.29 and h/h_imggray < 0.55:
            print(h,h_imggray,h/h_imggray)
            print(w,w_imggray,w/w_imggray)
            print("masuk if")
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
    print(the_charas)
    return img,erode1,the_charas,True

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def calcRealPosition(detections,net_width,net_height):
    width_real = 1920
    width_net = net_width

    height_real = 1080
    height_net = net_height

    for detection in detections:
        x, y, w, h = detection[2][0],detection[2][1],detection[2][2],detection[2][3] 
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        x, y, x1, y1 = xmin, ymin, xmax, ymax

        x_real = int( width_real - (width_real*(width_net - x)/width_net) )
        y_real = int( height_real - (height_real*(height_net - y)/height_net) )
        coor1 = (x_real,y_real)
        pojok_kiri_atas = [x_real,y_real]

        x1_real = int( width_real - (width_real*(width_net - x1)/width_net) )
        y1_real = int( height_real - (height_real*(height_net - y1)/height_net) )
        coor2 = (x1_real,y1_real)
        pojok_kanan_bawah = [x1_real,y1_real]
    try:
        return coor1, coor2, pojok_kiri_atas, pojok_kanan_bawah
    except:
        return 0, 0, 0, 0

netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolo-obj.cfg"
    weightPath = "backup/yolo-obj_2000.weights"
    metaPath = "./data/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("resources/Datasets/zzz-datatest/test4.MOV")
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    result_flag = False
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        try:
            h,w,depth = frame_read.shape
            toShow = frame_read.copy()
        except:
            break
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        coor1, coor2, pojok_kiri_atas, pojok_kanan_bawah = calcRealPosition(detections,darknet.network_width(netMain),darknet.network_height(netMain))
        if coor1 == 0:
            print(1/(time.time()-prev_time))
            cv2.imshow('Demo', frame_read)
            cv2.waitKey(3)
            continue
        temp_frame = toShow.copy()
        found_ROI = temp_frame[ pojok_kiri_atas[1]:pojok_kanan_bawah[1],pojok_kiri_atas[0]:pojok_kanan_bawah[0] ]
        newly, eroded, the_charas, seg_flag = segImg(found_ROI,"The file")
        charasnya = ''.join(the_charas)
        if len(the_charas)==0:
            charasnya = ""
        cv2.putText(toShow,'Detected Plate : '+charasnya,(20,100), font, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.rectangle(toShow, coor1, coor2, (0, 255, 0), 2)
        """
        if coor1 != 0:
            temp_frame = toShow.copy()
            found_ROI = temp_frame[ pojok_kiri_atas[1]:pojok_kanan_bawah[1],pojok_kiri_atas[0]:pojok_kanan_bawah[0] ]
            cv2.imwrite(network_path+'roi.jpg',found_ROI)
            
            theData = open(network_path+'theData.txt','w')
            theData.write('roi.jpg')
            theData.close()

            while True:
                result = open(network_path+'result.txt','r')
                try:
                    tempp = result.read()
                    if tempp=='0':
                        continue
                    elif '.jpg' in tempp:
                        resnya, ext = tempp.split('.')
                        result_flag = True
                        break
                    elif tempp == 'No Charas found':
                        result_flag = False
                        break
                    else:
                        result.close()
                        continue
                except:
                    result.close()

            
            cv2.rectangle(toShow, coor1, coor2, (0, 255, 0), 2)
            
            # cv2.rectangle(toShow, (20,20), (100,100), (0, 255, 0), 2)
            if result_flag == True:
                cv2.putText(toShow,'Detected Plate : '+resnya,(20,100), font, 1,(0,255,0),2,cv2.LINE_AA)
            elif result_flag == False:
                cv2.putText(toShow,'Detected Plate : NONE',(20,100), font, 1,(0,255,0),2,cv2.LINE_AA)
            result_flag = False
            # image = cvDrawBoxes(detections, frame_resized)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            print(1/(time.time()-prev_time))
            cv2.imshow('Demo', toShow)
            cv2.waitKey(3)
        elif coor1 == 0:
            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            theData = open(network_path+'theData.txt','w')
            theData.write('0')
            theData.close()
            
            print(1/(time.time()-prev_time))
            cv2.imshow('Demo', frame_read)
            cv2.waitKey(3)
        """
        
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        # continue
        # image = cvDrawBoxes(detections, frame_resized)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', toShow)
        cv2.waitKey(3)
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
