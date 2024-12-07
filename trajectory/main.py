from time import sleep

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
            if val is None:
                pos = i
                val = distance(point1, point2)
            else:
                if distance(point1, point2) < val:
                    pos = i
                    val = distance(point1, point2)
        order.append([point1, remaining[pos]])
        remaining.pop(pos)

    return order

    # current = centers[0]
    # order = [current]
    # remaining = centers[1:]
    #
    # while remaining:
    #     dist = None
    #     pos = None
    #     for i, point in enumerate(remaining):
    #         if pos is None:
    #             pos = i
    #             dist = distance(current, point)
    #         else:
    #             if distance(current, point) < dist:
    #                 dist = distance(current, point)
    #                 pos = i
    #     order.append(remaining[pos])
    #     current = remaining[pos]
    #     remaining.pop(pos)
    # return order


directory = "files/out/"
files = os.listdir(directory)

# sort files
length = len(files)
files_sort = [0]*length
for file in files:
    id = int(file[2:-4])
    files_sort[id] = file
print(files_sort)

image = None
image_last = None

plt.figure()
for file in files_sort:
    # save last image
    if image is None:
        image = np.load(directory + file)

    else:
        image_last = image

        print(f"{file}")
        image = np.load(directory+file)
        centers = get_move_points(image)
        centers_last = get_move_points(image_last)
        print(centers)
        print(centers_last)

        order = set_orders(centers, centers_last)
        print(order)
        order = np.array(order)

        for points in order:
            plt.plot(points[:,1], points[:,0], "r")

plt.show()

