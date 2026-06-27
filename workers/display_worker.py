import logging
import threading
from queue import Empty

import cv2
import numpy as np

from config import CAMERA_URLS
from utils.display import annotate_frame
from utils.queues import SlidingQueue

logger = logging.getLogger(__name__)


def display_worker(
    display_queues: dict[str, SlidingQueue],
    stop_event: threading.Event,
    annotated: bool = True,
):
    latest_frames: dict[str, np.ndarray | None] = {}

    for camera_name in CAMERA_URLS.keys():
        latest_frames[camera_name] = None

    try:
        while not stop_event.is_set():
            for camera_name, _ in CAMERA_URLS.items():
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

            if not all(latest_frames[name] is not None for name in CAMERA_URLS.keys()):
                stop_event.wait(0.01)
                continue

            top_row = np.hstack([latest_frames["cam1"], latest_frames["cam2"]])
            bottom_row = np.hstack([latest_frames["cam3"], latest_frames["cam4"]])
            grid = np.vstack([top_row, bottom_row])

            cv2.imshow("Camera Grid", grid)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Shutdown requested by user. Shutting down display thread.")
                stop_event.set()

    except Exception:
        # TODO: Implement retry/restart logic for the thread in case it crashes
        logger.exception("Display worker crashed.")
        stop_event.set()

    finally:
        cv2.destroyAllWindows()
        logger.info("Display worker stopped.")
