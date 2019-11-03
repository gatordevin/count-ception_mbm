from skimage.io import imread
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.misc
import os
import torch
from model import ModelCountception
import torch.multiprocessing as mp
import threading
import time
from threading import Lock
lock = Lock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelCountception().to(device)
model.eval()
print("Loading weights...")
from_before = torch.load('checkpoints/after_99_epochs.model')
model_weights = from_before['model_weights']
model.load_state_dict(model_weights)
outputTemplate = None
names = os.listdir("/home/techgarage/Downloads/ZStackImages/")
names.sort()
def TiffPreProcessing(path):
    im = imread(path)
    #print("Image array is: {0}".format(im))

    imageShape = im.shape
    imageXSize = imageShape[0]
    imageYSize = imageShape[1]
    #print("Image X size is: {0} and Image Y size is: {1}".format(imageXSize,imageYSize))

    xTargetSize = math.ceil(imageXSize/256)*256
    yTargetSize = math.ceil(imageYSize/256)*256
    #print(xTargetSize-imageXSize)
    #print(yTargetSize-imageYSize)
    im = np.pad(im,((0,xTargetSize-imageXSize),(0,yTargetSize-imageYSize)),'constant')

    imageShape = im.shape
    imageXSize = imageShape[0]
    imageYSize = imageShape[1]
    #print("Image X size is: {0} and Image Y size is: {1}".format(imageXSize,imageYSize))
    im = np.stack((im,)*3, axis=-1)
    return(im,imageXSize,imageYSize)


def ImageSegmenter(img):
    subImages = {}
    for x in range(int(img.shape[0]/256)):
            for y in range(int(img.shape[1]/256)):
                #print(x,y)
                subImages["{0},{1}".format(x,y)] = img[x*256:(x+1)*256,y*256:(y+1)*256]
    return(subImages)


def PytorchPreProcessing(img):
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    img = img.to(device)
    return img


def StichOutputs2D(outputTemplate,modelOutput,position):
    modelOutput = modelOutput[0].cpu().detach().numpy()
    result = np.concatenate(modelOutput, axis=1)[16:272,16:272]
    result[result<2] = 0
    x = int(position.split(",")[0])
    y = int(position.split(",")[1])
    outputTemplate[x*256:(x+1)*256,y*256:(y+1)*256] = result
    return(outputTemplate)


def analyzeBigImage():
    global names
    while names != None:
        lock.acquire()
        filename = names[0]
        print(filename)
        tiffImg, imWidth, imHeight = TiffPreProcessing("/home/techgarage/Downloads/ZStackImages/"+filename)
        names.pop(0)
        lock.release()
        outputTemplate = np.zeros([imWidth,imHeight],dtype=np.uint8)
        subImages = ImageSegmenter(tiffImg)
        for pos, img in subImages.items():
            print(pos)
            modelReadyImg = PytorchPreProcessing(img)
            output = model.forward(modelReadyImg)
            outputTemplate = StichOutputs2D(outputTemplate,output,pos)
            try:
                os.mkdir('/home/techgarage/Downloads/ZStackImages/analyzed/')
            except:
                "Already Exists"
        scipy.misc.imsave('/home/techgarage/Downloads/ZStackImages/analyzed/{0}.png'.format(filename.replace(".tif","")), outputTemplate)

t1 = threading.Thread(target=analyzeBigImage) 
t2 = threading.Thread(target=analyzeBigImage)
t1.start()
t2.start()
t1.join()
t2.join()
    
    

    
        
        