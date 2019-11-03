from skimage.io import imread
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
import matplotlib.pyplot as plt
import numpy as np
from tvtk.api import tvtk
from mayavi import mlab
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
final = None
count = 0
for filename in os.listdir("/home/techgarage/Downloads/ZStackImages/analyzed/"):
    count+=1
    print(count)
    if count == 1:
        im = imread("/home/techgarage/Downloads/ZStackImages/analyzed/" + filename)
        y = scipy.misc.imresize(im, [100,100], mode='L', interp='nearest')
        low_values_flags = y < 10
        y[low_values_flags] = 0
        final = y
    elif count == 25:
        break
    else:
        im = imread("/home/techgarage/Downloads/ZStackImages/analyzed/" + filename)
        y = scipy.misc.imresize(im, [100,100], mode='L', interp='nearest')
        low_values_flags = y < 10
        y[low_values_flags] = 0
        print(len(np.nonzero(y)[0]))
        final = np.dstack((final,y))

fig = plt.figure()
x,y,z = final.nonzero()
ax = fig.gca(projection='3d')
ax.scatter(x, y, -z, zdir='z', c= final[x,y,z])
mlab.points3d(final)
mlab.colorbar(orientation='vertical')
mlab.show()
plt.imshow(y)
plt.show()