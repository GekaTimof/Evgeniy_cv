import numpy as np
import cv2

winnameOrigin = "Image Origin"
cv2.namedWindow(winnameOrigin, cv2.WINDOW_NORMAL)
winnameGray = "Image Gray"
cv2.namedWindow(winnameGray, cv2.WINDOW_NORMAL)
winnameReturn = "Image Returned"
cv2.namedWindow(winnameReturn, cv2.WINDOW_NORMAL)

print("Start transforming")
dest_pts = np.float32([[250.0, 50.0],[675.0,40.0], [680.0, 350.0], [250.0,350.0]])
rect_width = 800
rect_height = 600
src_pts = np.float32([[0.0, 0.0], [rect_width, 0.0], [rect_width, rect_height], [0.0, rect_height]])

M = cv2.getPerspectiveTransform(dest_pts, src_pts)


def get_window(frame):
    # Преобразование изображения в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Диапазоны для тёмнозелёного цвета (H: 30-85, S: 50-255, V: 50-255)
    # Эти значения могут быть настроены в зависимости от желаемого оттенка зелёного
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Маска для выделения всех тёмнозелёных пикселей
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Применение маски к изображению
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Находим контуры всех зелёных областей
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отображаем найденные контуры и их центры
    for contour in contours:
        # Вычисление центра каждого контура
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Отображаем центр каждого зелёного объекта
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)  # Красный круг на центре

    # Показываем изображение с найденными зелёными точками
    cv2.imshow("Green Points", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_img(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    warped = cv2.warpPerspective(gray, M, (rect_width, rect_height))

    mean_dark_color = np.mean(warped[warped < np.mean(warped)])
    _, img_binary = cv2.threshold(warped, mean_dark_color, 255, cv2.THRESH_BINARY)

    height, width = img_binary.shape
    height_size = 5
    width_size = 5

    # opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_size, height_size))
    img_opened = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_open)

    # closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width_size, 2 * height_size))
    img_closed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel_close)

    img_inverted = cv2.bitwise_not(img_opened)
    return img_inverted


def return_img(warped):
    M_inv = np.linalg.inv(M)
    restored = cv2.warpPerspective(warped, M_inv, (frame.shape[1], frame.shape[0]))
    return restored


# get image from camera
vcap = cv2.VideoCapture("rtsp://192.168.43.1:8080/h264_ulaw.sdp")
ret, frame = vcap.read()
if not vcap.isOpened():
    print("Error: Cannot open RTSP stream.")
    exit()

# height, width = frame.shape[:2]
# crop_height = int(height * 0.5)
# crop_width = int(width * 0.4)
#
# start_x = (width - crop_width) // 2
# start_y = (height - crop_height) // 2
# frame = frame[start_y:start_y+crop_height, start_x:start_x+crop_width]


print("get image")
cv2.imshow(winnameOrigin, frame)
print(frame.shape)


get_window(frame)

# get warped image
warped = get_img(frame)
cv2.imshow(winnameGray, warped)

# return image
returned = return_img(warped)
cv2.imshow(winnameReturn, returned)


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vcap.release()
cv2.destroyAllWindows()