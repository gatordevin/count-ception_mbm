from tvtk.api import tvtk
import numpy as np
from mayavi import mlab
from numpy import random

X, Y, Z = np.mgrid[-10:10:100j, -10:10:100j, -10:10:100j]
data = np.random.random((40, 40,40))

x, y, z, value = np.random.random((4, 40,40))
mlab.points3d(data)
print(data.shape)
mlab.colorbar(orientation='vertical')
mlab.show()