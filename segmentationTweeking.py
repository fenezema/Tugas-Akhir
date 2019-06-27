#IMPORT
from ValidationPreprocess import *
#IMPORT

def screenShooter():
    fvs = FileVideoStream('D:\\05111540000055_PBaskara\\src\\resources\\Datasets\\zzz-datatest\\test3.mov',queue_size=128).start()
    counter = 0
    while fvs.more():
        frame = fvs.read()
        cv2.imshow('frame',frame)

        if cv2.waitKey(0) & 0xFF == ord('s'):
            cv2.imwrite('resources/segImgTweeking/'+str(counter)+'.jpg',frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        
        counter+=1
    
    cv2.destroyAllWindows()
    fvs.stop()

def doMorph(imgBin,filename):
    kernel = np.ones((3,3),np.uint8)
    erode = cv2.erode(imgBin,kernel,iterations = 1)
    dilate = cv2.dilate(erode,kernel,iterations = 2)
    erode1 = cv2.erode(dilate,kernel,iterations = 1)

    cv2.imwrite('resources/segImgTweeking/morph'+filename,erode1)

def segImg(filename):
    img = cv2.imread('resources/segImgTweeking/'+filename,0)
    pre = ValidationPreprocess()
    binImg = pre.imageToBinary(redefine={'flag':True,'img':img})

    cv2.imshow('res',binImg)
    doMorph(binImg,filename)
    cv2.imwrite('resources/segImgTweeking/res'+filename,binImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def main():
    args = sys.argv
    cmd_ind = args.index('--command') + 1
    cmd = args[cmd_ind]

    if cmd == 'ssFrame':
        screenShooter()
    elif cmd == 'segTweeking':
        filename_ind = args.index('--filename') + 1
        filename = args[filename_ind]
        segImg(filename)


if __name__=="__main__":
    main()