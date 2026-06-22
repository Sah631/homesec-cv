# Entrypoint for app
# Should initialise model and start video processing loop
import argparse
import logging

from models.yolo26 import model
from video import process_videos


logger = logging.getLogger(__name__)


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

    parser.add_argument(
        "--pre-roll-seconds",
        type=float,
        default=15.0,
        help="Seconds of pre-roll to include in each clip (default: 15.0)",
    )

    parser.add_argument(
        "--post-roll-seconds",
        type=float,
        default=15.0,
        help="Seconds of post-roll to include in each clip (default: 15.0)",
    )

    parser.add_argument(
        "--save-clips",
        action="store_true",
        default=False,
        help="Whether to save clips to disk (default: False, set to True to enable)",
    )

    parser.add_argument(
        "--clip-output-mode",
        choices=["annotated", "raw"],
        default="annotated",
        help="Whether to save raw frames or annotated frames with detections (default: annotated)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO"],
        default="INFO",
        help="Logging verbosity level (default: INFO)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting HomeSec CV App")

    yolo_model = model
    process_videos(
        yolo_model,
        fps=args.fps,
        pre_roll_seconds=args.pre_roll_seconds,
        post_roll_seconds=args.post_roll_seconds,
        save_clips=args.save_clips,
        clip_output_mode=args.clip_output_mode,
    )


if __name__ == "__main__":
    main()
