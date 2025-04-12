from ultralytics import YOLO
from pathlib import Path
import cv2
import time

data_folder = Path(__file__).parent / "data"
model_path = data_folder / "best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture("/dev/video0")
state = "idle" #wait, result
prev_time = 0
curr_time = 0

player1_hand = ""
player2_hand = ""
game_result = ""
timer_fix = 3
timer = 0
while cap.isOpened():
    ret, frame = cap.read()
    cv2.putText(frame, f"{state} - {max(0, (timer_fix - timer)):.1f} {game_result}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Camera", frame)
    results = model(frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    if state == "wait" or state == "result":
        timer = round(time.time() - prev_time, 1)

    if timer >= timer_fix * 2:
        state = "idle"
        game_result = ""
        timer = 0

    if not results:
        continue

    result = results[0]
    if result.boxes.xyxy.tolist():
        if len(result.boxes.xyxy) == 2:
            labels = []
            for label, xyxy in zip(result.boxes.cls, result.boxes.xyxy):
                x1, y1, x2, y2 = xyxy.cpu().numpy().astype("int")
                print(result.boxes.cls)
                labels.append(result.names[label.item()].lower())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255 ,0), 4)
                cv2.putText(frame, f"{labels[-1]}", (x1 + 20, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            player1_hand, player2_hand = labels

            if player1_hand == "rock" and player2_hand == "rock" and state == "idle":
                state = "wait"
                prev_time = time.time()

        if timer >= timer_fix:
            if state == "wait":
                if player1_hand == player2_hand:
                    game_result = "draw"
                elif ((player1_hand == "scissor" and player2_hand == "paper")
                      or (player1_hand == "rock" and player2_hand == "scissors")
                      or (player1_hand == "paper" and player2_hand == "rock")):
                    game_result = "player 1 win"
                elif ((player2_hand == "scissor" and player1_hand == "paper")
                      or (player2_hand == "rock" and player1_hand == "scissors")
                      or (player2_hand == "paper" and player1_hand == "rock")):
                    game_result = "player 2 win"
                else:
                    game_result = "err"
                state = "result"

    cv2.imshow("YOLO", frame)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
