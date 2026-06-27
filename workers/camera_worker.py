import logging
import threading
import time

import cv2

from config import CLIP_OUTPUT_FPS, DEFAULT_DIMENSIONS

# from video.frame_buffer import FrameBuffer
from schemas.packets import FramePacket
from utils.queues import SlidingQueue

logger = logging.getLogger(__name__)


# TODO: Implement automatic reconnection to camera if error is encountered
def camera_worker(
    camera_name: str,
    camera_url: str,
    stop_event: threading.Event,
    frame_queue: SlidingQueue,
    inference_fps: float = CLIP_OUTPUT_FPS,
    dims: tuple[int, int] = DEFAULT_DIMENSIONS,
):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)

    # TODO: Implement retry logic with backoff to make program more robust to temporary network issues or NVR restarts
    if not cap.isOpened():
        logger.error("Could not open RTSP stream for %s", camera_name)
        return

    logger.debug("Opened stream for %s", camera_name)

    # frame_buffer = FrameBuffer(pre_roll_seconds=pre_roll_seconds)
    inference_interval = 1.0 / inference_fps
    last_inference_time = 0
    frame_idx = 0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.warning("Failed to grab frame for camera %s", camera_name)
                break

            frame_idx += 1
            now = time.time()

            if now - last_inference_time < inference_interval:
                continue

            frame = cv2.resize(frame, dims, interpolation=cv2.INTER_LINEAR)

            frame_packet = FramePacket(
                camera_name=camera_name,
                timestamp=now,
                frame_index=frame_idx,
                frame=frame,
            )

            frame_queue.put_nowait(frame_packet)
            last_inference_time = now

    except Exception:
        logger.exception("Camera worker crashed for %s", camera_name)

    finally:
        logger.info("Releasing stream for %s", camera_name)
        cap.release()
