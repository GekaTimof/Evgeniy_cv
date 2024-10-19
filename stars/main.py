import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening

def count_figures(image, imageM, struct):
    count = 0
    Y = struct.shape[0] // 2
    X = struct.shape[1] // 2

    for y in range(Y, imageM.shape[0] - (Y+1)):
        for x in range(X, imageM.shape[1] - (X-1)):
            if image[y][x] == 1:
                sub = image[y - Y:y + (Y+1), x - X:x + (X+1)]
                if np.all(sub == struct):
                    count += 1
    return count

struct_plus = np.array([[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]])
struct_star = np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]])

image = np.load("files/starsnpy.txt")

plt.figure()
plt.subplot(131)
plt.imshow(image)

image_stars = binary_erosion(image, struct_star)
plt.subplot(132)
plt.imshow(image_stars)
count_stars = count_figures(image=image, imageM=image_stars, struct=struct_star)
print(f"count of stars = {count_stars}")

image_pluses = binary_erosion(image, struct_plus)
plt.subplot(133)
plt.imshow(image_pluses)
count_pluses = count_figures(image=image, imageM=image_pluses, struct=struct_plus)
print(f"count of pluses = {count_pluses}")

print(f"sum of pluses and stars = {count_stars + count_pluses}")

plt.show()
