import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import os

def get_move_points(image):
    labeled = label(image)
    regions = regionprops(labeled)
    centers = []
    for region in regions:
        cy, cx = region.centroid
        centers.append((int(cy), int(cx)))
    return centers


def distance(c1, c2):
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5


def set_orders(centers, centers_last):
    if not centers or not centers_last:
        return []

    if len(centers) != len(centers_last):
        return []

    order = []
    remaining = centers_last[:]

    for point1 in centers:
        val = None
        pos = None
        for i, point2 in enumerate(remaining):
            dist = distance(point1, point2)
            if val is None:
                pos = i
                val = dist
            else:
                if  dist< val:
                    pos = i
                    val = dist
        order.append([point1, remaining[pos]])
        remaining.pop(pos)

    return order

directory = "files/out/"
files = os.listdir(directory)

# sort files
length = len(files)
files_sort = [0]*length
for file in files:
    id = int(file[2:-4])
    files_sort[id] = file
print(files_sort)

centers = None
image = None

plt.figure()
for file in files_sort:
    # save last image
    if image is None:
        image = np.load(directory + file)
        centers = get_move_points(image)

    else:
        centers_last = centers

        print(f"{file}")
        image = np.load(directory+file)
        centers = get_move_points(image)
        # centers_last = get_move_points(image_last)
        print(centers)
        print(centers_last)

        order = set_orders(centers, centers_last)
        print(order)
        order = np.array(order)

        for points in order:
            plt.plot(points[:,1], points[:,0], "r")

plt.show()

