import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, euler_number
from collections import defaultdict
from pathlib import Path
from scipy.ndimage import binary_dilation


def recognize(region):
    if region.image.mean() == 1.0:
        return "-"
    else:
        image = region.image.copy()
        struct = [[0,1,0],[1,1,1],[0,1,0]]
        open_image = binary_dilation(image, struct)
        enumber = euler_number(open_image)
        if enumber == -1: # B or 8
            have_vl = np.sum(np.mean(region.image[:, :region.image.shape[1] // 2],0) == 1) > 3
            if have_vl:
                return "B"
            else:
                return "8"

        elif enumber == 0: # A, 0, P or D
            have_vl = np.sum(np.mean(region.image[:, :region.image.shape[1] // 2], 0) == 1) > 3
            if have_vl: # P, D
                top_sum = region.image[:, :region.image.shape[1] * 2 // 3].sum() / (region.image.size * 2 // 3)
                bot_sum = region.image[:, region.image.shape[1] * 2 // 3:].sum() / (region.image.size / 3)
                # print(top_sum, bot_sum)
                # print(np.isclose(top_sum, bot_sum, 0.2))
                if np.isclose(top_sum, bot_sum, 0.2):
                    return "D"
                else:
                    return "P"

            else:   # A, 0
                cy, cx = region.centroid_local
                cy /= region.image.shape[0]
                cx /= region.image.shape[1]
                # print(cy, cx)
                if cy < 0.5:
                    return "0"
                else:
                    return "A"
        else: # /, W, X, * or 1
            have_vl = np.sum(np.mean(region.image, 0) == 1) > 3
            if have_vl:
                return "1"
            else:
                if region.eccentricity < 0.45:
                    return "*"
                else:
                    image = region.image.copy()
                    image[0, :] = 1
                    image[-1, :] = 1
                    image[:, 0] = 1
                    image[:, -1] = 1
                    enumber = euler_number(image)
                    if enumber == -1:
                        return "/"
                    elif enumber == -3:
                        return "X"
                    elif enumber == -4:
                        return "W"

    return "@"

# test img
template = plt.imread('files/alphabet_ext.png')[:,:,:3].mean(2)
template[template < 1] = 0
template = np.logical_not(template)
template_labeled = label(template)
# real img
alphabet = plt.imread('files/symbols.png')[:,:,:3].mean(2)
alphabet[alphabet > 0] = 1
alphabet_labeled = label(alphabet)

regions = regionprops(alphabet_labeled)

# result = defaultdict(lambda: 0)
# for region in regions:
#     symbol = recognize(region)
#     result[symbol] += 1
#
# print(result)

symbols = defaultdict(lambda: 0)
path = Path("images")
path.mkdir(exist_ok=True)
plt.figure()
for i, region in enumerate(regions):
    print(region.label)
    symbol = recognize(region)
    symbols[symbol] += 1
    plt.cla()
    plt.title(f"Symbol - {symbol}")
    plt.imshow(region.image)
    plt.savefig(path/ f"image_{i}.png")

print(symbols)


# plt.figure()
# plt.subplot(211)
# plt.imshow(template_labeled)
# plt.show()