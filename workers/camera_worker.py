import logging
import threading
import time

import cv2

from config import CLIP_OUTPUT_FPS, DEFAULT_DIMENSIONS

# from video.frame_buffer import FrameBuffer
from schemas.packets import FramePacket
from utils.clips import FrameBuffer
from utils.queues import SlidingQueue

logger = logging.getLogger(__name__)


def _open_capture(camera_name: str, camera_url: str) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        logger.warning("Could not open RTSP stream for %s", camera_name)
        cap.release()
        return None

    logger.info("Opened stream for %s", camera_name)
    return cap

# TODO: Test stream fps and optimise this worker to improve throughput
def camera_worker(
    camera_name: str,
    camera_url: str,
    stop_event: threading.Event,
    frame_queue: SlidingQueue,
    frame_buffer: FrameBuffer,
    inference_fps: float = CLIP_OUTPUT_FPS,
    dims: tuple[int, int] = DEFAULT_DIMENSIONS,
    initial_reconnect_delay: float = 1.0,
    max_reconnect_delay: float = 30.0,
):
    """
    Capture frames from one camera stream and publish sampled frames for inference.

    Reconnects with backoff when the stream cannot be opened or frame capture
    fails, and exits when shutdown is requested through stop_event.
    """
    logger.info("Camera worker started for %s", camera_name)

    if inference_fps <= 0:
        raise ValueError("inference_fps must be > 0")

    # frame_buffer = FrameBuffer(pre_roll_seconds=pre_roll_seconds)
    inference_interval = 1.0 / inference_fps
    last_inference_time = 0
    frame_idx = 0
    reconnect_delay = initial_reconnect_delay

    while not stop_event.is_set():
        cap = _open_capture(camera_name=camera_name, camera_url=camera_url)

        if cap is None:
            logger.warning(
                "Retrying connection for %s in %.1f seconds.",
                camera_name,
                reconnect_delay,
            )
            stop_event.wait(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
            continue

        reconnect_delay = initial_reconnect_delay

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()

                if not ret or frame is None:
                    logger.warning(
                        "Failed to grab frame for camera %s. Reconnecting...",
                        camera_name,
                    )
                    break

                frame_idx += 1
                now = time.time()

                if now - last_inference_time < inference_interval:
                    continue

                frame = cv2.resize(frame, dims, interpolation=cv2.INTER_LINEAR)

                frame_buffer.add_frame(frame=frame, timestamp=now)

                frame_packet = FramePacket(
                    camera_name=camera_name,
                    timestamp=now,
                    frame_index=frame_idx,
                    frame=frame,
                )

                frame_queue.put_nowait(frame_packet)
                last_inference_time = now

        except Exception:
            logger.exception(
                "Unexpected error in camera worker for %s. Reconnecting...", camera_name
            )

        finally:
            logger.info("Releasing stream for %s", camera_name)
            cap.release()

        if not stop_event.is_set():
            logger.info(
                "Camera %s reconnecting in %.1f seconds.", camera_name, reconnect_delay
            )
            stop_event.wait(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    logger.info("Camera worker stopped for %s", camera_name)
