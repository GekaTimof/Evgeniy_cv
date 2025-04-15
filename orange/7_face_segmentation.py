import numpy as np
from ultralytics import YOLO
from pathlib import Path
import cv2
import time
from skimage import draw

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
# cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

data_folder = Path(__file__).parent / "data"
model_path = data_folder / "facial_best.pt"
model = YOLO(model_path)

# image = cv2.imread(data_folder / "me.jpg")
cap = cv2.VideoCapture("/dev/video0")

oranges = cv2.imread(data_folder / "oranges.png")
hsv_oranges = cv2.cvtColor(oranges, cv2.COLOR_BGR2HSV)

lower = (10, 240, 200)
upper = (20, 255, 255)
mask = cv2.inRange(hsv_oranges, lower, upper)
mask = cv2.dilate(mask, np.ones((7, 7)))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_contours = sorted(contours, key=cv2.contourArea)
m = cv2.moments(sorted_contours[-1])
print(m)
cx = int(m["m10"] / m ["m00"])
cy = int(m["m01"] / m ["m00"])

bbox = cv2.boundingRect(sorted_contours[-1])
# cv2.imshow("M", mask)

while cap.isOpened():
    ret, image = cap.read()

    result = model(image)[0]
    if result.boxes.xyxy.tolist():
        if len(result.boxes.xyxy) == 6:
            masks = result.masks
            global_mask = masks[0].data.cpu().numpy()[0, :, :]
            for mask in masks[1:]:
                global_mask += mask.data.cpu().numpy()[0, :, :]

            # cv2.imshow("Image", annotated)
            global_mask = cv2.resize(global_mask, (image.shape[1], image.shape[0])).astype(np.uint8)

            rr, cc = draw.disk((5, 5), 5)
            struct = np.zeros((11, 11), np.uint8)
            struct[rr, cc] = 1

            global_mask = cv2.erode(global_mask, struct, iterations=2)
            global_mask = cv2.dilate(global_mask, struct, iterations=3)
            global_mask = global_mask.reshape(image.shape[0], image.shape[1], 1)
            parts = (global_mask * image).astype("uint8")

            pos = np.where(global_mask > 0)
            min_y, max_y = int(np.min(pos[0]) * 0.7), int(np.max(pos[0]) * 1.1)
            min_x, max_x = int(np.min(pos[1]) * 0.8), int(np.max(pos[1]) * 1.2)
            global_mask = global_mask[min_y:max_y, min_x:max_x]
            parts = parts[min_y:max_y, min_x:max_x]

            resized_parts = cv2.resize(parts, (bbox[2], bbox[3]))
            resized_mask = cv2.resize(global_mask, (bbox[2], bbox[3])) * 255

            x, y, w, h = bbox
            oranges_copy = np.copy(oranges)
            roi = oranges_copy[y:y+h, x:x+w]
            bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(resized_mask))
            combined = cv2.add(bg, resized_parts)
            oranges_copy[y:y+h, x:x+w] = combined

            cv2.imshow("Images", oranges_copy)
            # cv2.imshow("Mask", parts)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()

