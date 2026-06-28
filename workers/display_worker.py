import logging
import threading
from queue import Empty

import cv2
import numpy as np

from utils.display import annotate_frame
from utils.queues import SlidingQueue

logger = logging.getLogger(__name__)


def display_worker(
    display_queues: dict[str, SlidingQueue],
    stop_event: threading.Event,
    camera_names: list[str],
    annotated: bool = True,
    display_sleep_seconds: float = 0.01,
):
    """
    Consume detection packets and render the latest frame from each camera.

    Optionally draws detection annotations, displays the camera grid, and
    requests shutdown when the user closes the display with the quit key.
    """
    logger.info("Display worker started.")

    latest_frames: dict[str, np.ndarray | None] = {
        camera_name: None for camera_name in camera_names
    }

    try:
        while not stop_event.is_set():
            for camera_name in camera_names:
                try:
                    detection_packet = display_queues[camera_name].get_nowait()
                except Empty:
                    continue

                if annotated:
                    latest_frames[camera_name] = annotate_frame(
                        detection_packet=detection_packet
                    )
                else:
                    latest_frames[camera_name] = detection_packet.frame

            if not all(latest_frames[name] is not None for name in camera_names):
                stop_event.wait(display_sleep_seconds)
                continue

            # Add a util function to dynamically create grid instead of hardcoding
            top_row = np.hstack([latest_frames["cam1"], latest_frames["cam2"]])
            bottom_row = np.hstack([latest_frames["cam3"], latest_frames["cam4"]])
            grid = np.vstack([top_row, bottom_row])

            cv2.imshow("Camera Grid", grid)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Shutdown requested by user. Shutting down display thread.")
                stop_event.set()
                break

            stop_event.wait(display_sleep_seconds)

    except Exception:
        logger.exception("Display worker crashed.")
        stop_event.set()

    finally:
        cv2.destroyAllWindows()
        logger.info("Display worker stopped.")
