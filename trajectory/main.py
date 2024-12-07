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


def set_order(centers):
    if not centers:
        return []

    current = centers[0]
    order = [current]
    remaining = centers[1:]

    while remaining:
        dist = None
        pos = None
        for i, point in enumerate(remaining):
            if pos is None:
                pos = i
                dist = distance(current, point)
            else:
                if distance(current, point) < dist:
                    dist = distance(current, point)
                    pos = i
        order.append(remaining[pos])
        current = remaining[pos]
        remaining.pop(pos)
    return order


directory = "files/out/"
files = os.listdir(directory)

plt.figure()

for file in files:
    print(f"{file}")
    image = np.load(directory+file)
    centers = get_move_points(image)
    print(centers)
    order = set_order(centers)
    print(order)
    order = np.array(order)

    plt.plot(order[:,1], order[:,0], "r")
    plt.plot(order[:,1], order[:,0], "bo")

plt.show()

