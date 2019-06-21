# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:38:56 2019

@author: Chastine
"""

from ValidationPreprocess import *

fvs = FileVideoStream('D:\\05111540000055_PBaskara\\src\\resources\\Datasets\\zzz-datatest\\test3.mov',queue_size=2048).start()
fps = FPS().start()
while fvs.more():
    frame = fvs.read()
    try:
        cv2.imshow("Frame",frame)
        cv2.waitKey(1)
        fps.update()
    except:
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
fvs.stop()