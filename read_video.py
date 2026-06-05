import os
import numpy as np
from urllib.parse import quote

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

from config import CAMERA_URLS, DETECTION_CLASSES
from models.yolo26 import model

import time

import cv2


cam1_url = CAMERA_URLS["cam1"]
cam2_url = CAMERA_URLS["cam2"]
cam3_url = CAMERA_URLS["cam3"]
cam4_url = CAMERA_URLS["cam4"]

cap1 = cv2.VideoCapture(cam1_url, cv2.CAP_FFMPEG)
cap2 = cv2.VideoCapture(cam2_url, cv2.CAP_FFMPEG)
cap3 = cv2.VideoCapture(cam3_url, cv2.CAP_FFMPEG)
cap4 = cv2.VideoCapture(cam4_url, cv2.CAP_FFMPEG)

if not cap1.isOpened():
    raise RuntimeError("Could not open RTSP stream for camera 1")

if not cap2.isOpened():
    raise RuntimeError("Could not open RTSP stream for camera 2")

if not cap3.isOpened():
    raise RuntimeError("Could not open RTSP stream for camera 3")

if not cap4.isOpened():
    raise RuntimeError("Could not open RTSP stream for camera 4")

uniform_dims = (640, 360)

last_inference_time = 0
latest_results = [None, None, None, None]

while True:
    ret, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    if not ret or not ret2 or not ret3 or not ret4:
        print("Failed to grab frame")
        break

    frame1 = cv2.resize(frame1, uniform_dims, interpolation=cv2.INTER_LINEAR)
    frame2 = cv2.resize(frame2, uniform_dims, interpolation=cv2.INTER_LINEAR)
    frame3 = cv2.resize(frame3, uniform_dims, interpolation=cv2.INTER_LINEAR)
    frame4 = cv2.resize(frame4, uniform_dims, interpolation=cv2.INTER_LINEAR)

    frames = [frame1, frame2, frame3, frame4]

    if time.time() - last_inference_time > 0.2:
        latest_results = model.predict(frames, save=False, conf=0.25, iou=0.45, imgsz=640, classes=DETECTION_CLASSES)
        last_inference_time = time.time()

    for result, frame in zip(latest_results, frames):
        if result is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    top_row = np.hstack([frame1, frame2])
    bottom_row = np.hstack([frame3, frame4])
    grid = np.vstack([top_row, bottom_row])

    cv2.imshow('Camera Grid', grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
