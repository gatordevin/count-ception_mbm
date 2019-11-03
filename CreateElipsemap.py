import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.patches import Ellipse
import csv
ells = []
with open('/home/techgarage/Downloads/1-117ROI.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if row[2] != "XM":
            ells.append([Ellipse((int(float(row[2])), 6000-int(float(row[3]))), int(float(row[4])), int(float(row[5])), int(float(row[6]))),int(float(row[1]))])
    
    a = plt.subplot(111, aspect='equal')
    cmap = matplotlib.cm.get_cmap('inferno')
    
    for e in ells:
        print(e)
        e[0].set_clip_box(a.bbox)
        e[0].set_alpha(0.5)
        a.add_artist(e[0])
        e[0].set_facecolor(cmap(e[1]/200))

    plt.xlim(0, 6000)
    plt.ylim(0, 6000)

    plt.show()  