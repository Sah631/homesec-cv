# Entrypoint for app
# Should initialise model and start video processing loop
import argparse

from models.yolo26 import model
from video import process_videos


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HomeSec-VC: Home Security Video Classifier"
    )

    # Command line arguments
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frames per second for processing video streams (default: 20.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    yolo_model = model
    process_videos(yolo_model, fps=args.fps)


if __name__ == "__main__":
    main()
