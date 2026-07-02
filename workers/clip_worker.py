import logging
import threading
import time
from queue import Empty, Queue

from utils.clips import ClipManager, FrameBuffer

logger = logging.getLogger(__name__)


def clip_worker(
    stop_event: threading.Event,
    detection_queue: Queue,
    clip_manager: ClipManager,
    frame_buffers: dict[str, FrameBuffer],
    queue_log_interval_seconds: float = 10.0,
):
    # Get detections from detection_queue, process with clip_manager, then write to clip_queue
    logger.info("Clip worker started.")

    last_queue_log_time = time.time()

    queue_max = detection_queue.maxsize

    while not stop_event.is_set():
        now = time.time()

        queue_size = detection_queue.qsize()

        if now - last_queue_log_time >= queue_log_interval_seconds:
            logger.info(
                "Detection queue depth: %d/%d %.0f%%",
                queue_size,
                queue_max,
                100 * queue_size / queue_max,
            )
            last_queue_log_time = now

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

        # Ensure clip manager logic is as expected. It should not write to file, it should return a clipwriter object if clip has ended
        # If it returns, then I will wrap it in a ClipPacket (or FilePacket), and add it to the file_queue (or clip_queue) to be processed
        # and saved by another worker
        clip_manager.process_frame(
            camera=detection_packet.camera_name,
            frame=detection_packet.frame,
            frame_buffer=frame_buffers[detection_packet.camera_name],
            detections=detection_packet.interesting_detections,
        )
    pass
