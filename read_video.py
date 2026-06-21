import os
import numpy as np
import json
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

from config import CAMERA_URLS, DETECTION_CLASSES, IGNORE_ZONES, SAVE_COOLDOWN
from models.yolo26 import model

import time

import cv2

CAMERA_NAMES = ["cam1", "cam2", "cam3", "cam4"]
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
METADATA_PATH = DATA_DIR / "metadata" / "detections.jsonl"


def ensure_data_dirs():
    for camera_name in CAMERA_NAMES:
        (RAW_DATA_DIR / camera_name).mkdir(parents=True, exist_ok=True)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_detections(result):
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append(
            {
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return detections


def center_inside(bbox, zone):
    x1, y1, x2, y2 = bbox
    zone_x1, zone_y1, zone_x2, zone_y2 = zone
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2


def is_detection_ignored(camera_name, detection):
    for ignore_zone in IGNORE_ZONES.get(camera_name, []):
        ignored_class_ids = ignore_zone.get("class_ids")

        if (
            ignored_class_ids is not None
            and detection["class_id"] not in ignored_class_ids
        ):
            continue

        if center_inside(detection["bbox"], ignore_zone["bbox"]):
            return True

    return False


def save_detection_frame(camera_name, frame, detections):
    timestamp = datetime.now().astimezone()
    timestamp_for_file = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    image_path = RAW_DATA_DIR / camera_name / f"{camera_name}_{timestamp_for_file}.jpg"

    if not cv2.imwrite(str(image_path), frame):
        print(f"Failed to save frame: {image_path}")
        return None

    metadata = {
        "timestamp": timestamp.isoformat(),
        "camera": camera_name,
        "image_path": str(image_path),
        "detections": detections,
    }

    with METADATA_PATH.open("a", encoding="utf-8") as metadata_file:
        metadata_file.write(json.dumps(metadata) + "\n")

    return image_path


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

uniform_dims = (1280, 720)

last_inference_time = 0
last_save_time = [0, 0, 0, 0]
latest_results = [None, None, None, None]

ensure_data_dirs()

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
    clean_frames = [frame.copy() for frame in frames]

    now = time.time()

    if now - last_inference_time > 0.2:
        latest_results = model.predict(
            frames,
            save=False,
            conf=0.25,
            iou=0.45,
            imgsz=640,
            classes=DETECTION_CLASSES,
        )
        last_inference_time = now

        for camera_index, result in enumerate(latest_results):
            camera_name = CAMERA_NAMES[camera_index]
            detections = extract_detections(result)
            interesting_detections = [
                detection
                for detection in detections
                if not is_detection_ignored(camera_name, detection)
            ]

            if (
                interesting_detections
                and now - last_save_time[camera_index] > SAVE_COOLDOWN
            ):
                image_path = save_detection_frame(
                    camera_name, clean_frames[camera_index], interesting_detections
                )

                if image_path is not None:
                    last_save_time[camera_index] = now

    for result, frame in zip(latest_results, frames):
        if result is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    top_row = np.hstack([frame1, frame2])
    bottom_row = np.hstack([frame3, frame4])
    grid = np.vstack([top_row, bottom_row])

    cv2.imshow("Camera Grid", grid)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
