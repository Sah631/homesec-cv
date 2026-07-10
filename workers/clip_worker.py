import logging
import threading
from queue import Empty, Queue

from utils.clips import ClipManager, FrameBuffer

logger = logging.getLogger(__name__)


def clip_worker(
    stop_event: threading.Event,
    detection_queue: Queue,
    clip_manager: ClipManager,
    frame_buffers: dict[str, FrameBuffer],
):
    logger.info("Clip worker started.")

    try:
        while not stop_event.is_set():
            try:
                detection_packet = detection_queue.get_nowait()
            except Empty:
                stop_event.wait(0.01)  # Add variable for this
                continue

            if (
                detection_packet is None
                or detection_packet.frame_packet is None
                or detection_packet.frame is None
            ):
                logger.error("Detection packet malformed. Skipping for clip saving")
                continue

            clip_manager.process_frame(
                detection_packet=detection_packet,
                frame_buffer=frame_buffers[detection_packet.camera_name],
            )

    except Exception:
        logger.exception("Clip worker crashed.")
        stop_event.set()

    finally:
        logger.info("Clip worker stopped.")
