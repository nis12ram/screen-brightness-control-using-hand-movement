# Setting brightness

import cv2
import os
from ultralytics import YOLO
import numpy as np
import screen_brightness_control as sbc

# here the video path on which you want to run the model
video_path = ''

# name the path of handdetection.pt file (available in repo)
model_path = ''

model = YOLO(model_path)  # load a custom mode

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    height = frame.shape[0]
    width = frame.shape[1]

    if ret == False:
        break

    image_array = np.array(frame)
    results = model(image_array)
    boxes = results[0].boxes.data.tolist()

    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        norm_cx = int(cx / width * 100)
        norm_cy = int(cy / height * 100)

        # animating height
        width = (norm_cy * 480) / 100

        # Setting brightness
        sbc.set_brightness(norm_cy)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.rectangle(frame, (7, 7), (38, int(width)), (200, 255, 255), -1)

    cv2.rectangle(frame, (5, 5), (40, 470), (125, 0, 0), 2)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
