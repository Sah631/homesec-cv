import cv2
from config import CAMERA_URLS, DETECTION_CLASSES
import time
import numpy as np
from ultralytics import YOLO


def open_video_streams() -> dict[str, cv2.VideoCapture]:
    caps = {}
    for camera_name, url in CAMERA_URLS.items():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        # TODO: Implement retry logic with backoff to make program more robust to temporary network issues or NVR restarts
        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream for {camera_name}")
        caps[camera_name] = cap
    
    return caps

def resize_frames(frames: list, target_dims: tuple = (1280, 720)) -> list:
    return [cv2.resize(frame, target_dims, interpolation=cv2.INTER_LINEAR) for frame in frames]

def process_videos(model: YOLO) -> None:
    # Video processing loop
    caps = open_video_streams()
    last_inference_time = 0

    while True:
        ret1, frame1 = caps["cam1"].read()
        ret2, frame2 = caps["cam2"].read()
        ret3, frame3 = caps["cam3"].read()
        ret4, frame4 = caps["cam4"].read()

        if not ret1 or not ret2 or not ret3 or not ret4:
            print("Failed to grab frame")
            break
        
        uniform_dims = (1280, 720)

        frames = resize_frames([frame1, frame2, frame3, frame4], uniform_dims)

        now = time.time()

        if now - last_inference_time > 0.05:
            latest_results = model.predict(
                frames,
                save=False,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                classes=DETECTION_CLASSES,
                verbose=False,
            )
            last_inference_time = now

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

    for cap in caps.values():
        cap.release()
        
    cv2.destroyAllWindows()
