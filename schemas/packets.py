from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FramePacket:
    """Dataclass for items sent from cameras to the inference worker"""

    camera_name: str
    timestamp: float
    frame_index: int
    frame: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.frame.shape

    @property
    def height(self) -> int:
        return self.frame.shape[0]

    @property
    def width(self) -> int:
        return self.frame.shape[1]


@dataclass(slots=True)
class Detection:
    """Model agnostic class for storing model predictions"""

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(slots=True)
class DetectionPacket:
    """Dataclass for items sent from inference worker"""

    frame_packet: FramePacket
    detections: list[Detection]
    inference_timestamp: float
    inference_latency_ms: float
    model_name: str

    @property
    def camera_name(self) -> str:
        return self.frame_packet.camera_name

    @property
    def frame(self) -> np.ndarray:
        return self.frame_packet.frame
