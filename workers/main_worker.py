# Owns queue creation, main process execution, and ending/closing all threads
import logging
from threading import Event, Thread

from config import CAMERA_URLS
from detectors.yolo26 import YOLODetector
from utils.queues import SlidingQueue
from workers import camera_worker, display_worker, inference_worker

logger = logging.getLogger(__name__)


def main_worker():
    """
    Main orchestrator thread responsible for creating all background threads, owning queues for passing data
    between threads, and exiting the program cleanly.
    """
    frame_queues: dict[str, SlidingQueue] = {}
    # detection_queue = SlidingQueue(maxsize=10)
    display_queues: dict[str, SlidingQueue] = {}
    camera_threads: dict[str, Thread] = {}
    stop_event = Event()
    detector = YOLODetector(model_name="YOLO26")

    for camera_name, camera_url in CAMERA_URLS.items():
        frame_queues[camera_name] = SlidingQueue(
            maxsize=1
        )  # Should make this a constant

        display_queues[camera_name] = SlidingQueue(maxsize=1)

        # Need to pass specific camera queue as parameter to this worker
        camera_threads[camera_name] = Thread(
            target=camera_worker,
            name=f"CameraWorker-{camera_name}",
            kwargs={
                "camera_name": camera_name,
                "camera_url": camera_url,
                "stop_event": stop_event,
                "frame_queue": frame_queues[
                    camera_name
                ],  # Add fps and dims as kwargs later
            },
        )

    # Pass all frame_queues as parameter to this worker
    inference_thread = Thread(
        target=inference_worker,
        name="InferenceWorker",
        kwargs={  # Once inference_worker is created, ensure the kwargs match
            "detector": detector,
            "frame_queues": frame_queues,
            "display_queues": display_queues,
            "stop_event": stop_event,
        },
    )

    # Pass display_queues as parameter to this worker
    display_thread = Thread(
        target=display_worker,
        name="DisplayWorker",
        kwargs={  # Once display_worker is created, ensure the kwargs match
            "display_queues": display_queues,
            "stop_event": stop_event,
        },
    )

    all_threads = [
        *camera_threads.values(),
        inference_thread,
        display_thread,
    ]

    try:
        logger.info("Starting worker threads.")

        for thread in all_threads:
            thread.start()

        for thread in all_threads:
            thread.join()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Requesting shutdown.")
        stop_event.set()

        for thread in all_threads:
            logger.info("Waiting for thread %s to stop.", thread.name)
            thread.join(timeout=5.0)

    finally:
        stop_event.set()
        logger.info(
            "All worker threads stopped or shutdown requested. Exiting program."
        )
