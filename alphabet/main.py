import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops

def fill(binary):
    result = binary.copy()
    for l, row in enumerate(binary):
        switch = np.sum(row[1:] != row[:-1]) // 2 * 2 - 1
        # print(switch)
        mark = False
        for i in range(1, len(row) - 1):
            if row[i-1] == 1 and row[i] == 0 and switch > 0:
                switch -= 1
                mark = True

            if row[i-1] == 0 and row[i] == 1 and switch > 0:
                mark = False
                switch -= 1

            if mark:
                result[l][i] = 1
    return result

image = plt.imread("files/lama_on_moon.png")
image = image[80:-40, 60:-40]

gray = np.mean(image, 2)
conts = sobel(gray)
thresh = threshold_otsu(conts)
binary = conts > thresh
labeled = label(binary)
for region in regionprops(labeled):
    if region.area < 200 or region.perimeter < 1000:
        binary[np.where(labeled == region.label)] = 0

binary = fill(binary)

plt.figure()
plt.imshow(binary, cmap="gray")
plt.show()

