import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2

from config import IGNORE_ZONES


RAW_DATA_DIR = Path("data") / "raw"


def latest_image_for_camera(camera_name):
    camera_dir = RAW_DATA_DIR / camera_name
    image_paths = sorted(
        path
        for path in camera_dir.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {camera_dir}")

    return image_paths[-1]


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview configured ignore zones on a saved frame."
    )
    parser.add_argument(
        "--camera", default="cam1", help="Camera name from config.py, e.g. cam1"
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Image path to preview. Defaults to latest camera image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image or latest_image_for_camera(args.camera)

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    preview = draw_ignore_zones(image, args.camera)
    window_title = f"Ignore zones: {args.camera} - {image_path}"

    print(f"Previewing {image_path}")
    cv2.imshow(window_title, preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
