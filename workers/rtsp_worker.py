import logging
import subprocess
import threading
from queue import Empty

import cv2
import numpy as np

from config import (
    CAMERA_URLS,
    RTSP_DISPLAY_FRAME_DIMENSIONS,
    RTSP_DISPLAY_GRID_DIMENSIONS,
    RTSP_OUTPUT_URL,
)
from utils.queues import SlidingQueue

logger = logging.getLogger(__name__)


CELL_WIDTH = RTSP_DISPLAY_FRAME_DIMENSIONS[0]
CELL_HEIGHT = RTSP_DISPLAY_FRAME_DIMENSIONS[1]
GRID_WIDTH = RTSP_DISPLAY_GRID_DIMENSIONS[0]
GRID_HEIGHT = RTSP_DISPLAY_GRID_DIMENSIONS[1]


def start_ffmpeg_rtsp_process(
    output_url: str,
    width: int,
    height: int,
    fps: int,
) -> subprocess.Popen:
    command = [
        "ffmpeg",
        "-loglevel",
        "warning",
        # Raw frames coming from Python stdin
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        # Low-latency H.264 encoding
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        # Publish to RTSP
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        output_url,
    ]

    logger.info("Starting FFmpeg RTSP publisher")

    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def rtsp_worker(
    latest_annotated_frames: dict[str, SlidingQueue],
    stop_event: threading.Event,
    output_fps: float = 10.0,
):
    logger.info("RTSP grid worker started.")

    if output_fps <= 0:
        raise ValueError("output_fps must be > 0")

    camera_names = list(CAMERA_URLS.keys())

    if len(camera_names) != 4:
        raise ValueError("RTSP grid worker currently expects exactly 4 cameras.")

    black_frame = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)

    frames: list[np.ndarray] = [
        black_frame.copy(),
        black_frame.copy(),
        black_frame.copy(),
        black_frame.copy(),
    ]

    process = start_ffmpeg_rtsp_process(
        output_url=RTSP_OUTPUT_URL,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        fps=output_fps,
    )

    if process.stdin is None:
        raise RuntimeError("FFmpeg stdin is not available.")

    frame_interval = 1.0 / output_fps

    try:
        while not stop_event.is_set():
            for i, camera_name in enumerate(camera_names):
                try:
                    frame = latest_annotated_frames[camera_name].get_nowait()
                except Empty:
                    logger.debug("No latest annotated frame for %s", camera_name)
                    continue

                if frame is None:
                    continue

                if frame.shape[:2] != (CELL_HEIGHT, CELL_WIDTH):
                    frame = cv2.resize(
                        frame,
                        (CELL_WIDTH, CELL_HEIGHT),
                        interpolation=cv2.INTER_LINEAR,
                    )

                frames[i] = frame

            if not all(frame is not None for frame in frames):
                stop_event.wait(0.01)
                continue

            top_row = np.hstack([frames[0], frames[1]])
            bottom_row = np.hstack([frames[2], frames[3]])
            grid = np.ascontiguousarray(np.vstack([top_row, bottom_row]))

            if process.poll() is not None:
                logger.error("FFmpeg process exited unexpectedly.")
                stop_event.set()
                break

            try:
                process.stdin.write(grid.tobytes())
                process.stdin.flush()
            except BrokenPipeError:
                logger.exception("FFmpeg pipe broke. Stopping RTSP grid worker.")
                stop_event.set()
                break

            stop_event.wait(frame_interval)
    except Exception:
        logger.exception("RTSP grid worker crashed.")
        stop_event.set()

    finally:
        logger.info("Stopping RTSP grid worker.")

        if process.stdin:
            process.stdin.close()

        process.terminate()

        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg did not terminate cleanly. Killing process.")
            process.kill()

        logger.info("RTSP grid worker stopped.")
