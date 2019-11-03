
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tvtk.api import tvtk
from mayavi import mlab
import os
count = 0
xArr = []
yArr = []
zArr = []
sArr = []
zeroes = np.zeros((6400,6400,3), dtype=int)
for filename in os.listdir("/home/techgarage/Downloads/ZStackImages/analyzed/"):
    count += 1
    print(count)
    if count < 40:
        finalContour = []
        im = cv2.imread("/home/techgarage/Downloads/ZStackImages/analyzed/" + filename)
        imBlack = im
        imBlack[imBlack<5] = 0
        imBlack[imBlack>5] = 255
        imgray = cv2.cvtColor(imBlack,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        thresh = cv2.GaussianBlur(thresh, (3,3), 0)
        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        for countour in contours:
            if(countour.size>200):
                finalContour.append(countour)
        
        
        for countour in finalContour:
            for upperCountour in countour:
                for x,y in upperCountour:
                    xArr.append(int(x))
                    yArr.append(int(y))
                    zArr.append(count*2)
                    sArr.append(3)
                    zeroes[int(x),int(y)] = (255,255,255)
        #print(xArr)
        
    else:
        break
mlab.points3d(xArr,yArr,zArr,sArr,scale_factor=2)
mlab.show()
plt.imshow(zeroes)
plt.show()