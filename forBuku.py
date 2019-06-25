#IMPORT
from ValidationPreprocess import *
import sys
#IMPORT

def doBab541(path, filename):
    img = cv2.imread(path+filename,0)
    cv2.imwrite('hasilnya_grayscale.jpg',img)
    pre = ValidationPreprocess()
    imgBin = pre.imageToBinary(redefine={'flag':True,'img':img})
    cv2.imwrite('hasilnya_binary.jpg',imgBin)

    imgBinResized = pre.imageToBinary(redefine={'flag':True,'img':img},resizeImg=True,sizeImg=32)
    cv2.imwrite('hasilnya_binary_resized.jpg',imgBinResized)

def doBab542(path, filename):
    img = cv2.imread(path+filename,0)
    cv2.imwrite('hasilnya542_grayscale.jpg',img)
    pre = ValidationPreprocess()
    imgBin = pre.imageToBinary(redefine={'flag':True,'img':img})
    cv2.imwrite('hasilnya542_binary.jpg',imgBin)

    kernel = np.ones((3,3),np.uint8)
    erode = cv2.erode(imgBin,kernel,iterations = 1)
    cv2.imwrite('hasilnya542_erode.jpg',erode)
    dilate = cv2.dilate(erode,kernel,iterations = 2)
    cv2.imwrite('hasilnya542_dilate.jpg',dilate)
    erode1 = cv2.erode(dilate,kernel,iterations = 1)
    cv2.imwrite('hasilnya542_erode1.jpg',erode1)

if __name__=="__main__":
    args = sys.argv

    try:
        path_ind = args.index('--path') + 1
        filename_ind = args.index('--filename') + 1
        fungsi_ind = args.index('--method') + 1

        path = args[path_ind]
        filename = args[filename_ind]
        fungsi = args[fungsi_ind]

        if fungsi == 'doBab541':
            doBab541(path,filename)
        elif fungsi == 'doBab542':
            doBab542(path,filename)
    except:
        print('Wrong usage')

