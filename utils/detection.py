from config import CLASS_NAMES
from schemas.packets import Detection


def convert_yolo_to_detection(result) -> list[Detection]:
    detections: list[Detection] = []

    if result is None or result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
        confidence = float(box.conf[0])

        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                xyxy=(x1, y1, x2, y2),
            )
        )

    return detections
