from collections import defaultdict
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv

def is_ball(region):
    if region.eccentricity < 0.45:
        return True
    else:
        return False

image = plt.imread("files/balls_and_rects.png")

image_hsv = rgb2hsv(image)[:,:,0]
# print(image_hsv)

binary = image_hsv > 0
binary[binary > 0] = 1
labeled = label(binary)
print(np.max(labeled))

regions = regionprops(labeled)

# hsv value for each region
regions_colors = {}
colors_all = []
for y in range(image_hsv.shape[0]):
    for x in range(image_hsv.shape[1]):
        if labeled[y][x] > 0:
            regions_colors[int(labeled[y][x])] = image_hsv[y][x]
            colors_all.append(float(image_hsv[y][x]))

colors_all = np.array(list(set(colors_all)))
colors_all.sort()
colors_all = colors_all[::-1]

# get hsv value for color changing
dif_vals = []
for i, elem in enumerate(colors_all[:-1] - colors_all[1:] > 0.004):
    if elem > 0:
        dif_vals.append(float(colors_all[i]))
dif_vals.append(0)
print(dif_vals)


# set color number for each region
for key in regions_colors.keys():
    for i in range(len(dif_vals)):
        if regions_colors[key] >= dif_vals[i]:
            regions_colors[key] = i
            break

# calculate count for each figure and for each combination of color and figure
color_figures_count = defaultdict(lambda: 0)
for i, region in enumerate(regions):
    if is_ball(region):
        color_figures_count['ball'] += 1
        color_figures_count[f'ball-{regions_colors[i+1]}'] += 1
    else:
        color_figures_count['rect'] += 1
        color_figures_count[f'rect-{regions_colors[i+1]}'] += 1

for key in sorted(color_figures_count.keys()):
    print(f"{key} - {color_figures_count[key]}")

plt.figure()
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.plot(colors_all, "o")
plt.show()
