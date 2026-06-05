from dotenv import load_dotenv
import os

load_dotenv()


USERNAME = os.environ["HV_USER"]
PASSWORD = os.environ["HV_PW"]
NVR_IP = os.environ["NVR_IP"]
PORT = os.environ["PORT"]

RTSP_URL_BASE = f"rtsp://{USERNAME}:{PASSWORD}@{NVR_IP}:{PORT}/Streaming/Channels/"

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
    14, # bird
    15, # cat
    16, # dog
]
