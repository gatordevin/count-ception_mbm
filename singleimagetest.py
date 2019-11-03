from skimage.io import imread
import torch
from model import ModelCountception
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelCountception().to(device)
model.eval()
print("Loading weights...")
from_before = torch.load('checkpoints/after_49_epochs.model')
model_weights = from_before['model_weights']
model.load_state_dict(model_weights)

for filename in os.listdir("/home/techgarage/count-ception_mbm/utils/Test/"):
    im = imread("/home/techgarage/count-ception_mbm/utils/Test/" + filename)
    print(im.shape)
    if(len(im.shape)==2):
        img = np.stack((im,)*3, axis=-1)
    else:
        img = im
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    img = img.to(device)
    output = model.forward(img)
    patch_size = 32
    ef = ((patch_size / 1) ** 2.0)
    output_count = (output.cpu().detach().numpy() / ef).sum(axis=(2, 3))
    print(output_count)
    output_arr = output[0].cpu().detach().numpy()
    #print(np.concatenate(output_arr, axis=1)[16:272,16:272].shape)
    plt.imshow(im)
    plt.imshow(np.concatenate(output_arr, axis=1)[16:272,16:272],alpha=0.5)
    plt.show()
    plt.savefig('test_outputs/samples_{0}'.format(filename))