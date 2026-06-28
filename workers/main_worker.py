# Owns queue creation, main process execution, and ending/closing all threads
import logging
from threading import Event, Thread

from config import CAMERA_URLS
from detectors.yolo26 import YOLODetector
from utils.queues import SlidingQueue
from workers import camera_worker, display_worker, inference_worker

logger = logging.getLogger(__name__)


# TODO: Pass in arguments from main (like annotated display, detector, etc.)
def main_worker():
    """
    Start and supervise the camera, inference, and display workers.

    Creates the shared queues and shutdown event, launches all worker threads,
    watches for unexpected worker exits, and joins threads during shutdown.
    """
    frame_queues: dict[str, SlidingQueue] = {}
    # detection_queue = SlidingQueue(maxsize=10)
    display_queues: dict[str, SlidingQueue] = {}
    camera_threads: dict[str, Thread] = {}
    stop_event = Event()
    detector = YOLODetector(model_name="YOLO26")
    camera_names = list(CAMERA_URLS.keys())

    for camera_name, camera_url in CAMERA_URLS.items():
        frame_queues[camera_name] = SlidingQueue(
            maxsize=1
        )  # Should make this a constant

        display_queues[camera_name] = SlidingQueue(maxsize=1)

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

    inference_thread = Thread(
        target=inference_worker,
        name="InferenceWorker",
        kwargs={
            "detector": detector,
            "frame_queues": frame_queues,
            "display_queues": display_queues,
            "stop_event": stop_event,
        },
    )

    display_thread = Thread(
        target=display_worker,
        name="DisplayWorker",
        kwargs={
            "display_queues": display_queues,
            "stop_event": stop_event,
            "camera_names": camera_names,
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

        while not stop_event.is_set():
            for thread in all_threads:
                if not thread.is_alive():
                    logger.warning("Thread %s exited.", thread.name)
                    stop_event.set()
                    break
            
            stop_event.wait(1.0)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Requesting shutdown.")
        stop_event.set()
    
    except Exception:
        logger.exception("Main worker crashed. Requesting shutdown.")
        stop_event.set()

    finally:
        stop_event.set()

        for thread in all_threads:
            logger.info("Waiting for thread %s to stop.", thread.name)
            thread.join(timeout=5.0)

        logger.info(
            "All worker threads stopped or shutdown requested. Exiting program."
        )
