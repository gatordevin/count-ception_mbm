from skimage.io import imread
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.misc

im = imread("/home/techgarage/Downloads/00500.tif")
print("Image array is: {0}".format(im))

imageShape = im.shape
imageXSize = imageShape[0]
imageYSize = imageShape[1]
print("Image X size is: {0} and Image Y size is: {1}".format(imageXSize,imageYSize))

xTargetSize = math.ceil(imageXSize/256)*256
yTargetSize = math.ceil(imageYSize/256)*256
print(xTargetSize-imageXSize)
print(yTargetSize-imageYSize)
im = np.pad(im,((0,xTargetSize-imageXSize),(0,yTargetSize-imageYSize)),'constant')

imageShape = im.shape
imageXSize = imageShape[0]
imageYSize = imageShape[1]
print("Image X size is: {0} and Image Y size is: {1}".format(imageXSize,imageYSize))
im = np.stack((im,)*3, axis=-1)
for x in range(int(xTargetSize/256)):
    for y in range(int(yTargetSize/256)):
        print(x,y)
        scipy.misc.imsave('ImageSegments/x:{0},y:{1}.PNG'.format(x,y), im[x*256:(x+1)*256,y*256:(y+1)*256])