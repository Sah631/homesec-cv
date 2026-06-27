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
):
    logger.info("Inference worker started.")

    while not stop_event.is_set():
        # 2. fetch frame from each frame queue

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
            stop_event.wait(0.01)  # Add variable for this
            continue

        frames = [packet.frame for packet in frame_packets]

        # 3. run inference
        inference_start_time = time.time()

        try:
            latest_results = detector.predict(
                frames,
                save=False,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                classes=DETECTION_CLASSES,
                verbose=False,
            )
        except Exception:
            logger.exception("Inference failed")
            stop_event.wait(0.1)
            continue

        inference_end_time = time.time()
        batch_latency_ms = (inference_end_time - inference_start_time) * 1000.0
        per_frame_latency_ms = batch_latency_ms / max(len(frame_packets), 1)

        # 4. convert to detectionpacket
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

            # 5. push to detection queue
            # detection_queue.put_nowait(detection_packet)
