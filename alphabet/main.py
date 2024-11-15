import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, perimeter
from collections import defaultdict
from pathlib import Path


def distance(v1, v2):
    return ((v1 - v2) ** 2).sum() ** 0.5

def extractor(region):
    area = np.sum(region.image) / region.image.size
    perimeter_value = region.perimeter / region.image.size
    # cy, cx = region.centroid_local
    # cy /= region.image.shape[0]
    # cx /= region.image.shape[1]
    euler = region.euler_number
    eccentricity = region.eccentricity
    have_vl = np.sum(np.mean(region.image,0)) > 3
    # new metrics
    aspect_ratio = region.image.shape[1] / region.image.shape[0]
    compactness = (4 * np.pi * region.area) / (region.perimeter or 1) ** 2
    left_switch = sum(region.image[:,0][1:] != region.image[:,0][:-1])
    # print(left_switch)
    vertical_center_switch = sum(region.image[:,region.image.shape[1] // 2][1:] != region.image[:,region.image.shape[1] // 2][:-1])
    # print(vertical_center_switch)

    return np.array([area, perimeter_value, euler, eccentricity, have_vl, aspect_ratio, compactness, left_switch, vertical_center_switch])


def classificator(region, classes):
    det_class = None
    v = extractor(region)
    min_d = 10 ** 16
    for cls in classes:
        d = distance(v, classes[cls])
        if d < min_d:
            det_class = cls
            min_d = d
    return det_class

template = plt.imread('files/alphabet-small.png')[:,:,:3].mean(2)
template[template < 1] = 0
template = np.logical_not(template)

template_labeled = label(template)
template_count = np.unique(template_labeled)[-1]
print(f"template_count = {template_count}")

regions = regionprops(template_labeled)
classes = {"8": extractor(regions[0]),
           "0": extractor(regions[1]),
           "A": extractor(regions[2]),
           "B": extractor(regions[3]),
           "1": extractor(regions[4]),
           "W": extractor(regions[5]),
           "X": extractor(regions[6]),
           "*": extractor(regions[7]),
           "/": extractor(regions[8]),
           "-": extractor(regions[9])
           }

alphabet = plt.imread('files/alphabet.png')[:,:,:3].mean(2)
alphabet[alphabet > 0] = 1

alphabet_labeled = label(alphabet)
alphabet_count = np.unique(alphabet_labeled)[-1]
print(f"alphabet_count = {alphabet_count}")

# show classes
plt.figure()
# for i, key  in enumerate(classes.keys()):
#     plt.subplot(2,5,i+1)
#     plt.title(f'{i} -> {key} ({classes[key]},)')
#     plt.imshow(regions[i].image)



symbols = defaultdict(lambda: 0)
path = Path("images")
path.mkdir(exist_ok=True)
plt.figure()
for i, region in enumerate(regionprops(alphabet_labeled)):
    print(region.label)
    symbol = classificator(region, classes)
    symbols[symbol] += 1
    plt.cla()
    plt.title(f"Symbol - {symbol}")
    plt.imshow(region.image)
    plt.savefig(path/ f"image_{i}.png")

print(symbols)


# show raw data
# plt.figure()
# plt.subplot(211)
# plt.imshow(template)
#
# plt.subplot(212)
# plt.imshow(alphabet_labeled)
plt.show()






