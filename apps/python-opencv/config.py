import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


USERNAME = os.environ["HV_USER"]
PASSWORD = os.environ["HV_PW"]
NVR_IP = os.environ["NVR_IP"]
PORT = os.environ["PORT"]

RTSP_URL_BASE = f"rtsp://{USERNAME}:{PASSWORD}@{NVR_IP}:{PORT}/Streaming/Channels/"

MEDIAMTX_HOST = os.environ["MEDIAMTX_HOST"]
MEDIAMTX_PORT = os.environ["MEDIAMTX_PORT"]
MEDIAMTX_ANGRID_PATH = os.environ["MEDIAMTX_ANGRID_PATH"]

MEDIAMTX_PUBLISH_USER = os.environ["MEDIAMTX_PUBLISH_USER"]
MEDIAMTX_PUBLISH_PW = os.environ["MEDIAMTX_PUBLISH_PW"]

RTSP_OUTPUT_URL = f"rtsp://{MEDIAMTX_PUBLISH_USER}:{MEDIAMTX_PUBLISH_PW}@{MEDIAMTX_HOST}:{MEDIAMTX_PORT}/{MEDIAMTX_ANGRID_PATH}"

CAMERA_URLS = {
    "cam1": RTSP_URL_BASE + "102",
    "cam2": RTSP_URL_BASE + "202",
    "cam3": RTSP_URL_BASE + "302",
    "cam4": RTSP_URL_BASE + "402",
}

DETECTION_CLASSES = [
    0,  # person
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
    14,  # bird
    15,  # cat
    16,  # dog
]

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
}

SAVE_COOLDOWN = (
    1  # seconds between saving images of the same class from the same camera
)

# Clip General
SAVE_CLIPS_ENABLED = True

# Clip Timings
CLIP_PRE_ROLL_SECONDS = 10
CLIP_POST_ROLL_SECONDS = 5
CLIP_COOLDOWN_SECONDS = 2
CLIP_MAX_DURATION_SECONDS = 60

# Clip Info
CLIP_OUTPUT_FPS = 20.0
CLIP_OUTPUT_MODE = "annotated"
CLIP_LOCATION_MODE = "outside_ignore_zones"

# Clip Paths
CLIP_OUTPUT_DIR = Path("data/clips")
CLIP_METADATA_PATH = Path("data/metadata/clips.jsonl")

# Ignore zones use the same pixel coordinates as the frames passed to YOLO.
# Current read_video.py frame size is 1280x720, and each box is [x1, y1, x2, y2].
IGNORE_ZONES = {
    "cam1": [
        {
            "name": "parked_upper_driveway_car",
            "bbox": [110, 85, 845, 390],
            "class_ids": [2],  # car
        },
        {
            "name": "parked_lower_driveway_car",
            "bbox": [320, 225, 1120, 720],
            "class_ids": [2],  # car
        },
        {
            "name": "parked_opposite_road_left_car",
            "bbox": [670, 0, 780, 30],
            "class_ids": [2],  # car
        },
        {
            "name": "parked_opposite_road_cars",
            "bbox": [1030, 0, 1180, 75],
            "class_ids": [2],  # car
        },
    ],
}

# Frame Dimensions
DEFAULT_DIMENSIONS = (1280, 720)
RTSP_DISPLAY_FRAME_DIMENSIONS = (640, 360)
RTSP_DISPLAY_GRID_DIMENSIONS = (
    RTSP_DISPLAY_FRAME_DIMENSIONS[0] * 2,
    RTSP_DISPLAY_FRAME_DIMENSIONS[1] * 2,
)

MODEL_PATH = "checkpoints/yolo26m.pt"
