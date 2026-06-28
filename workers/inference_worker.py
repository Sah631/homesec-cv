import logging
import threading
import time
from queue import Empty

from config import DETECTION_CLASSES
from schemas.packets import DetectionPacket
from utils.detection import convert_yolo_to_detection
from utils.queues import SlidingQueue

logger = logging.getLogger(__name__)


def inference_worker(
    detector,
    frame_queues: dict[str, SlidingQueue],
    display_queues: dict[str, SlidingQueue],
    # detection_queue: SlidingQueue,
    stop_event: threading.Event,
    idle_sleep_seconds: float = 0.01,
    error_sleep_seconds: float = 0.1,
):
    """
    Consume frame packets, run batched detector inference, and publish detections.

    Records inference timing for each output packet and requests shutdown after
    repeated consecutive inference failures.
    """
    logger.info("Inference worker started.")

    failure_count = 0

    while not stop_event.is_set():
        frame_packets = []

        for queue in frame_queues.values():
            try:
                frame_packet = queue.get_nowait()
            except Empty:
                continue

            if frame_packet is None:
                continue

            frame_packets.append(frame_packet)

        if not frame_packets:
            stop_event.wait(idle_sleep_seconds)
            continue

        frames = [packet.frame for packet in frame_packets]

        inference_start_time = time.time()

        try:
            # TODO: Move model-specific args inside the detector class
            latest_results = detector.predict(
                frames,
                save=False,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                classes=DETECTION_CLASSES,
                verbose=False,
            )

            failure_count = 0
        except Exception:
            failure_count += 1
            logger.exception("Inference failed")

            if failure_count >= 5:
                logger.error("Inference failed %d consecutive times. Requesting shutdown.", failure_count)
                stop_event.set()
                break

            stop_event.wait(error_sleep_seconds)
            continue

        inference_end_time = time.time()
        batch_latency_ms = (inference_end_time - inference_start_time) * 1000.0
        per_frame_latency_ms = batch_latency_ms / max(len(frame_packets), 1)

        for frame_packet, result in zip(frame_packets, latest_results):
            detections = convert_yolo_to_detection(result)

            detection_packet = DetectionPacket(
                frame_packet=frame_packet,
                detections=detections,
                inference_timestamp=inference_end_time,
                inference_latency_ms=per_frame_latency_ms,
                model_name=detector.get_model_name(),
            )

            display_queues[frame_packet.camera_name].put_nowait(detection_packet)

    logger.info("Inference worker stopped.")
