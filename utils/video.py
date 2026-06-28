import cv2

from config import CLASS_NAMES, IGNORE_ZONES


def extract_detections(result) -> list[dict]:
    """Convert YOLO result object to list of dicts with class_id, class_name, confidence, and bbox"""
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"Unknown_{class_id}"),
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return detections


def center_inside(bbox, zone) -> bool:
    """Check if the center of the bbox is inside the given zone"""
    x1, y1, x2, y2 = bbox
    zone_x1, zone_y1, zone_x2, zone_y2 = zone
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2


def is_detection_ignored(camera_name, detection) -> bool:
    """Check if a detection is within an ignore zone"""
    for ignore_zone in IGNORE_ZONES.get(camera_name, []):
        ignored_class_ids = ignore_zone.get("class_ids")

        if (
            ignored_class_ids is not None
            and detection["class_id"] not in ignored_class_ids
        ):
            continue

        if center_inside(detection["bbox"], ignore_zone["bbox"]):
            return True

    return False


def draw_frame(result, frame):
    annotated_frame = frame.copy()

    if result is None:
        return annotated_frame

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = f"{CLASS_NAMES.get(cls, f'Unknown_{cls}')} {conf:.2f}"
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


# TODO: Make this function cleaner and integrate with annotate_frame in display.py
def draw_ignore_zones(image, camera_name):
    output = image.copy()

    for ignore_zone in IGNORE_ZONES.get(camera_name, []):
        x1, y1, x2, y2 = ignore_zone["bbox"]
        label = ignore_zone.get("name", "ignore_zone")

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

    return output
