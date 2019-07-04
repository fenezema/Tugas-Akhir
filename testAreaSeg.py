from DoYOLO import *

path = 'D:/dip_Official/src/Tugas-Akhir/resources/PositiveNew_v2/proc/foregrounddd4_frame.jpg'

img = cv2.imread(path)
imgNew,erode1,the_charas,flag = segImg(img,'')
print(the_charas)

hoii = ''.join(the_charas)

jei = hoii.split('-')
print(jei)
# h,w = img.shape
# exp_w = int(0.2692*w)
# first,second,third = img[:,:exp_w],img[:,(exp_w-1):(w-exp_w)],img[:,(w-exp_w):w]
# cv2.imshow('first',first)
# cv2.imshow('2nd',second)
cv2.imshow('3rd',erode1)

cv2.waitKey(0)
cv2.destroyAllWindows()