import matplotlib
from matplotlib.pyplot import figure
from numpy.ma.core import count

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

def count_figures(image, imageM, struct):
    count = 0
    Y = struct.shape[0] // 2
    X = struct.shape[1] // 2

    for y in range(Y, imageM.shape[0] - (Y+1)):
        for x in range(X, imageM.shape[1] - (X+1)):
            if imageM[y][x] == 1:
                sub = image[y - Y:y + (Y+1), x - X:x + (X+1)]
                if np.all(sub == struct):
                    count += 1
    return count


def rotate90(matrix):
        return np.array([list(reversed(col)) for col in zip(*matrix)])


struct_arr = [[4, [1,1,1,1,1,1,0],[1,1,1,1,1,1,0],[1,1,0,0,1,1,0],[1,1,0,0,1,1,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]],
              [2, [1,1,1,1,1,1,0],[1,1,1,1,1,1,0],[1,1,1,1,1,1,0],[1,1,1,1,1,1,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]]

image = np.load("files/ps.npy.txt").astype(int)

plt.figure()
plt.imshow(image)

label = two_pass(image)
print(f"count of all figures = {np.unique(label)[-1]} \n")

for struct_raw in struct_arr:
    struct = struct_raw[1:]
    for i in range(struct_raw[0]):
        count = 0
        struct = rotate90(struct)
        # print(f"struct: \n{struct}")

        image_erosion = binary_erosion(image, struct)

        count = count_figures(image=image, imageM=image_erosion, struct=struct)
        print(f"count of this figure = {count}")


