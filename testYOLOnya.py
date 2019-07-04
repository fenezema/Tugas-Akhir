from DoYOLO import *

for i in range(2):
    inp = input("hehe : ")
    img = cv2.imread('resources/GUIresources/saved_frames/'+inp)
    # img1 = cv2.imread('resources/GUIresources/saved_frames/test3-173.jpg')
    coor1, coor2, pojok_kiri_atas, pojok_kanan_bawah = YOLO(img)
    # coor11, coor21, pojok_kiri_atas1, pojok_kanan_bawah1 = YOLO(img1)
    cv2.rectangle(img,coor1,coor2,(0,255,0),2)
    # cv2.rectangle(img1,coor11,coor21,(0,255,0),2)
    cv2.imshow('res',img)
    # cv2.imshow('res',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()