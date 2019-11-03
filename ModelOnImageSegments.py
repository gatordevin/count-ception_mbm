from skimage.io import imread
import torch
from model import ModelCountception
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelCountception().to(device)
model.eval()
print("Loading weights...")
from_before = torch.load('checkpoints/after_49_epochs.model')
model_weights = from_before['model_weights']
model.load_state_dict(model_weights)
modeloutput = np.zeros([6656,6656],dtype=np.uint8)
originalImage = np.zeros([6656,6656,3],dtype=np.uint8)
count = 0
for filename in os.listdir("/home/techgarage/count-ception_mbm/ImageSegments/"):
    count+=1
    print(count)
    im = imread("/home/techgarage/count-ception_mbm/ImageSegments/" + filename)
    input = torch.from_numpy(im).float()
    input = input.permute(2, 0, 1).unsqueeze(0)
    input = input.to(device)
    output = model.forward(input)
    patch_size = 32
    ef = ((patch_size / 1) ** 2.0)
    output_count = (output.cpu().detach().numpy() / ef).sum(axis=(2, 3))
    #print(output_count)
    output_arr = output[0].cpu().detach().numpy()
    result = np.concatenate(output_arr, axis=1)[16:272,16:272]
    result[result<2] = 0
    #print(np.amax(result))
    x = int(filename.split(":")[1].split(",")[0])
    y = int(filename.split(":")[2].split(".")[0])
    modeloutput[x*256:(x+1)*256,y*256:(y+1)*256] = result
    originalImage[x*256:(x+1)*256,y*256:(y+1)*256] = im
    #plt.imshow(im)
    #plt.imshow(result,alpha=0.5)
    #empty[x,y] = output_count
    #plt.imshow(empty)
    #plt.show()
plt.imshow(originalImage)
plt.imshow(modeloutput,alpha=0.5)

plt.tight_layout()
plt.show()
plt.savefig('ImageSegmentResults/{0}'.format("Overall"))