import zmq
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from pathlib import Path
import cv2
import time


def angle(a, b, c):
    d = np.arctan2(c[1] - b[1], c[0] - b[0])
    e = np.arctan2(a[1] - b[1], a[0] - b[0])
    angle_ = np.rad2deg(d - e)
    angle_ = angle_ + 360 if angle_ < 0 else angle_
    return 360 - angle_ if angle_ > 180 else angle_

def process(image, keypoints):
    nose_seen = keypoints[0][0] > 0 and keypoints[0][1] > 0
    left_ear_seen = keypoints[3][0] > 0 and keypoints[0][1] > 0
    right_ear_seen = keypoints[4][0] > 0 and keypoints[0][1] > 0
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_hand = keypoints[9]
    right_hand = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    elbow = [0, 0]
    angle_elbow = -1
    try:
        if left_ear_seen and not right_ear_seen:
            angle_elbow = angle(left_shoulder, left_elbow, left_hand)
            elbow = left_elbow
        else:
            angle_elbow = angle(right_shoulder, right_elbow, right_hand)
            elbow = right_elbow

        x, y = int(elbow[0]) + 10, int(elbow[1]) + 10
        cv2.putText(image, f"{int(angle_elbow)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (25, 25, 255), 1)
        return int(angle_elbow)

    except ZeroDivisionError:
        return None

data_folder = Path(__file__).parent / "data"
model_path = data_folder / "yolo11n-pose.pt"

model = YOLO(model_path)

cv2.namedWindow("IMG", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture("/dev/video0")

last_time = time.time()
last_move = time.time()
flag = False
count = 0
writer = cv2.VideoWriter("videos/out.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 20, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()
    frame_counter = frame.copy()

    cure_time = time.time()
    cv2.putText(frame, f"FPS: {1 / (cure_time - last_time):.1f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (25, 255, 25), 1)
    last_time = cure_time

    results = model(frame)

    cv2.imshow("IMG", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    if not results:
        continue

    result = results[0]
    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue

    keypoints = keypoints[0]
    if not keypoints:
        continue


    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()

    angle_ = process(annotated, keypoints)
    if flag and angle_ > 150:
        flag = False
        count += 1
        last_move = time.time()
    elif not flag and angle_< 110:
        flag = True
        last_move = time.time()

    if time.time() - last_move  >= 10:
        count = 0

    cv2.putText(annotated, f"Count = {count}", (10, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1, (25, 25, 255),1)
    cv2.imshow("Pose", annotated)

    cv2.putText(frame_counter, f"Count = {count}", (10, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1, (25, 25, 255), 1)
    writer.write(frame_counter)

writer.release()
cap.release()
cv2.destroyAllWindows()
