import cv2

from schemas.packets import DetectionPacket


def annotate_frame(detection_packet: DetectionPacket):
    """Returns an annotated frame based on the detections in detection_packet"""
    annotated_frame = detection_packet.frame.copy()

    if detection_packet is None:
        return annotated_frame

    for detection in detection_packet.detections:
        x1, y1, x2, y2 = map(int, detection.xyxy)
        conf = detection.confidence
        label = f"{detection.class_name} {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated_frame
