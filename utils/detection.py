from config import CLASS_NAMES, IGNORE_ZONES
from schemas.packets import Detection


def center_inside(bbox, zone) -> bool:
    """Check if the center of the bbox is inside the given zone"""
    x1, y1, x2, y2 = bbox
    zone_x1, zone_y1, zone_x2, zone_y2 = zone
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2


def is_detection_ignored(camera_name: str, detection: Detection) -> bool:
    """Check if a detection is within an ignore zone"""
    for ignore_zone in IGNORE_ZONES.get(camera_name, []):
        ignored_class_ids = ignore_zone.get("class_ids")

        if (
            ignored_class_ids is not None
            and detection.class_id not in ignored_class_ids
        ):
            continue

        if center_inside(detection.xyxy, ignore_zone["bbox"]):
            return True

    return False


def convert_yolo_to_detection(
    camera_name, result
) -> tuple[list[Detection], list[Detection]]:
    detections: list[Detection] = []
    interesting_detections: list[Detection] = []

    if result is None or result.boxes is None:
        return detections, interesting_detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
        confidence = float(box.conf[0])

        detection = Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            xyxy=(x1, y1, x2, y2),
        )

        if not is_detection_ignored(camera_name=camera_name, detection=detection):
            interesting_detections.append(detection)

        detections.append(detection)

    return detections, interesting_detections
