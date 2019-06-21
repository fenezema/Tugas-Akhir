# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:50:37 2019

@author: fenezema
"""

#IMPORT
from ValidationPreprocess import *
from ModelBuild import *
#IMPORT

#GLOBAL INIT
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

# function to get the output layer names 
# in the architecture
def get_output_layers(net): 
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,255,0), 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

def processImage(image,index,WEIGHTS,CONFIG,scale, conf_threshold, nms_threshold,classes,COLORS,net):

    Width = image.shape[1]
    Height = image.shape[0]

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
            
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
    
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
    
    # display output image    
    out_image_name = "object detection"+str(index)
    #cv2.imshow(out_image_name, image)
    # wait until any key is pressed
    #cv2.waitKey()


def main():
    counter=0
    # 'path to yolo config file' 
    # download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.cfg
    CONFIG='resources/YOLOWeights/yolo-obj.cfg'
    
    # 'path to text file containing class names'
    # download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
    CLASSES='resources/YOLOWeights/obj.names'
    classes = None
    with open(CLASSES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    scale = 0.00392
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # 'path to yolo pre-trained weights' 
    # wget https://pjreddie.com/media/files/yolov3.weights
    WEIGHTS='resources/YOLOWeights/yolo-obj_2000.weights'
    
    templates = []
#    frames = cv2.VideoCapture("D:\\05111540000055_PBaskara\\src\\resources\\Datasets\\zzz-datatest\\test3.mov")
    fvs = FileVideoStream('resources/Datasets/zzz-datatest/test3.mov',queue_size=2048).start()

    # read pre-trained model and config file
    net = cv2.dnn.readNet(WEIGHTS, CONFIG)
    
    while fvs.more():
        frame = fvs.read()
        
#        # Our operations on the frame come here
        grayy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        row, col, dep = frame.shape
        processImage(frame,counter,WEIGHTS,CONFIG,scale,conf_threshold,nms_threshold,classes,COLORS,net)
    
        counter+=1
        
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