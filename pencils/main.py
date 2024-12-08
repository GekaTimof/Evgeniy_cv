import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import os
import cv2
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


def classificator(regions):
    match_percent = 10
    classes = {}
    classes_id = 1
    print("\nStart classification:")
    for id, region in regions:
        print(f"id = {id}")
        # print(f"classes = {classes}")

        v = extractor(region)
        if not classes:
            # contain all information about class
            classes[classes_id] = {}
            # contain information about all class elements - [count, [regions]]
            classes[classes_id][0] = [1, [region]]
            # contain information about class elements on one image (id - img name) - [count, [regions]]
            classes[classes_id][id] = [1, [region]]
            classes_id += 1
        else:
            not_in_classes = True
            keys = list(classes.keys())
            # try to add to exist class
            for key in keys:
                class_v = extractor(classes[key][0][1][0])
                # print(distance(v, class_v))
                # check class in classes
                if distance(v, class_v) <= match_percent:
                    # add to exist class
                    # add 1 to all count
                    classes[key][0][0] += 1
                    # add region to all regions
                    classes[key][0][1].append(region)

                    # check file (id)
                    if id in classes[key].keys():
                        # add 1 to id count
                        classes[key][id][0] += 1
                        # add region to id regions
                        classes[key][id][1].append(region)
                    else:
                        # contain information about class elements on one image (id - img name) - [count, [regions]]
                        classes[key][id] = [1, [region]]
                    not_in_classes = False

            # add new class
            if not_in_classes:
                # contain all information about class
                classes[classes_id] = {}
                # contain information about all class elements - [count, [regions]]
                classes[classes_id][0] = [1, [region]]
                # contain information about class elements on one image (id - img name) - [count, [regions]]
                classes[classes_id][id] = [1, [region]]
                classes_id += 1
    print("End classification\n")
    return classes


def show_classes(classes):
    for key in classes.keys():
        print(f"class {key}:")
        for id in classes[key]:
            if id == 0:
                print(f"total number of class elements = {classes[key][id][0]}")
            else:
                print(f"number of class elements in file - {id} = {classes[key][id][0]}")

def save_classes_img(classes, dir_name = "classes"):
    path = Path(dir_name)
    path.mkdir(exist_ok=True)
    for key in classes.keys():
        for id in classes[key]:
            for i, region in enumerate(classes[key][id][1]):
                plt.cla()
                plt.title(f"Class -{key}, file - {id}")
                plt.imshow(region.image)
                plt.savefig(path / f"Region_{i}.png")



# get files
directory = "files/images/"
files = os.listdir(directory)

# sort files
length = len(files)
files_sort = [0]*length
for file in files:
    id = int(file[5:-5]) - 1
    files_sort[id] = file
# print(files_sort)

# files without pencils
files_without_pencils = files_sort[:1]

# files with pencils
files_with_pencils = files_sort[1:]

# percent to stay only big regions
percent = 0.005

regions_marked = []
print(f"Get regions from files")
for file in files_sort[:5]:
    print(file)
    image = cv2.imread(directory + file)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=5)

    labeled = label(thresh)
    regions = regionprops(labeled)

    img_size = thresh.size
    # print(f"image size = {img_size}")
    for region in regions:
        # print(region.area)
        if region.area > img_size * percent:
            regions_marked.append([file, region])


regions_classified = classificator(regions_marked)
# print(regions_classified)
show_classes(regions_classified)
save_classes_img(regions_classified)