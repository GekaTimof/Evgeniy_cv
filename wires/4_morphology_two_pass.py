import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening

def neighbours2(x, y):
    return (y, x-1), (y-1, x)


def is_element_exist(array, x, y):
    return 0 <= x < array.shape[0] and 0 <= y < array.shape[1]

# get fild and neighbours arr, neighbour = (x:int, y:int)
# return arr with exist neighbours
def exist(B, nbs):
    exist_nbs = []
    for nb in nbs:
        x, y = nb
        if is_element_exist(B, x, y):
            exist_nbs.append(nb)
    return tuple(exist_nbs)


def get_neighbours_val(array, nbs):
    min = -1
    for nb in nbs:
        x, y = nb
        if array[x, y] != 0 and (min == -1 or array[x, y] < min):
            min = array[x, y]
    return min

def find(label, linked):
    j = label
    while linked[j] != 0:
        j = linked[j]
    return j

def union(label1, label2, linked):
    j = find(label1, linked)
    k = find(label2, linked)
    if j != k:
        linked[k] = j

def two_pass(B):
    B = (B.copy() * - 1).astype("int")
    linked = np.zeros(len(B), dtype="uint")
    LB = np.zeros_like(B)
    label = 1
    for y in range(B.shape[0]):
        for x in range(B.shape[1]):
            if B[y, x] != 0:
                nbs = neighbours2(x, y)
                exist_nbs = exist(B, nbs)
                nb_val = get_neighbours_val(LB, exist_nbs)

                if nb_val == -1:
                    m = label
                    label += 1
                elif nb_val > -1:
                    m = nb_val
                LB[y, x] = m

                for n in exist_nbs:
                    if LB[n] > 0:
                        lb = LB[n]
                        if lb != m:
                            union(m, lb, linked)

    for y in range(LB.shape[0]):
        for x in range(LB.shape[1]):
            if B[y, x] != 0:
                new_label = find(LB[y, x], linked)
                if new_label != LB[y, x]:
                    LB[y, x] = new_label


    uniq = np.unique(LB)[1:]
    for i, elem in enumerate(uniq):
        LB[LB == elem] = i+1


    return LB


def calculate_cuts(image, struct):
    plt.figure()
    plt.subplot(131)
    plt.imshow(image)

    label = two_pass(image)

    plt.subplot(132)
    plt.imshow(label)
    wires_count = np.unique(label)[-1]

    image_cut = binary_opening(image, struct)
    plt.subplot(133)
    plt.imshow(image_cut)

    for i in range(1, wires_count + 1):
        wire_image = label == i
        wire_cut = binary_opening(wire_image, struct)
        label_cut = two_pass(wire_cut)
        wires_parts = np.unique(label_cut)[-1]

        if wires_parts == 0:
            print(f"wire was destroyed")
        elif wires_parts == 1:
            print(f"wire wasn't cut")
        else:
            print(f"wire {i} was cuе for {wires_parts} parts")

    print()
    plt.show()


for file_n in range(1,7):
    struct = np.array([[0,1,0],[0,1,0],[0,1,0]])
    image = np.load(f"files/wires{file_n}npy.txt")
    calculate_cuts(image, struct)



