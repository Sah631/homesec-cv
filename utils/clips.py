import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2

from config import (
    CLIP_COOLDOWN_SECONDS,
    CLIP_MAX_DURATION_SECONDS,
    CLIP_METADATA_PATH,
    CLIP_OUTPUT_DIR,
    CLIP_OUTPUT_FPS,
    CLIP_OUTPUT_MODE,
    CLIP_POST_ROLL_SECONDS,
    CLIP_PRE_ROLL_SECONDS,
)
from schemas.packets import Detection

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Timestamped rolling frame buffer."""

    def __init__(self, pre_roll_seconds: float = CLIP_PRE_ROLL_SECONDS):
        self.pre_roll_seconds = pre_roll_seconds
        self.buffer = deque()
        self._lock = threading.Lock()

    def add_frame(self, frame, timestamp: float | None = None):
        timestamp = time.time() if timestamp is None else timestamp
        frame_to_store = frame.copy() if hasattr(frame, "copy") else frame

        with self._lock:
            self.buffer.append((timestamp, frame_to_store))
            self._trim_unlocked(timestamp)

    def trim(self, now: float | None = None):
        now = time.time() if now is None else now

        with self._lock:
            self._trim_unlocked(now)

    def _trim_unlocked(self, now: float):
        oldest_allowed = now - self.pre_roll_seconds

        while self.buffer and self.buffer[0][0] < oldest_allowed:
            self.buffer.popleft()

    def get_frames(self):
        with self._lock:
            return [frame for _, frame in self.buffer]

    def get_entries(self):
        with self._lock:
            return list(self.buffer)


@dataclass
class ClipEvent:
    camera: str
    clip_path: str
    writer: cv2.VideoWriter
    start_time: float
    last_detection_time: float
    trigger_detections: list[
        dict
    ]  # This is for metadata - stores info about detections. Should have the format: [{"class_id": int, "class_name": str, "confidence": float, "bbox": [x1, y1, x2, y2]}, ...]


class ClipManager:
    def __init__(
        self,
        cooldown_seconds: float = CLIP_COOLDOWN_SECONDS,
        post_roll_seconds: float = CLIP_POST_ROLL_SECONDS,
        max_duration_seconds: float = CLIP_MAX_DURATION_SECONDS,
    ):
        self.active_clips: dict[str, ClipEvent] = {}
        self.cooldown_seconds = cooldown_seconds
        self.post_roll_seconds = post_roll_seconds
        self.max_duration_seconds = max_duration_seconds
        self.last_clip_end_times: dict[str, float] = {}

    def process_frame(
        self, camera: str, frame, frame_buffer: FrameBuffer, detections: list | None
    ):
        """Main method to be called for each frame. Manages starting new clips based on detections, appending frames to active clips, and ending clips after post-roll period has passed without new detections."""
        event = self.active_clips.get(camera)

        now = time.time()

        # Makes sure to only start clip if there are detections (handles empty list)
        if detections:
            if event is None and self._cooldown_over(camera, now):
                self._start_clip(
                    camera, frame_buffer.get_entries(), now, detections
                )  # Create new ClipEvent and add to active_clips
                return

            if event is not None:
                event.last_detection_time = now

        event = self.active_clips.get(camera)

        if event is None:
            return

        try:
            event.writer.write(frame)  # Write frame to video if clip is active
        except Exception:
            logger.exception("Failed to write frame for camera %s", camera)

        if (
            now - event.last_detection_time > self.post_roll_seconds
            or now - event.start_time > self.max_duration_seconds
        ):
            self._end_clip(camera)  # Finalise and save clip, remove from active_clips

        return

    def _start_clip(
        self,
        camera: str,
        pre_roll_entries: list[tuple[float, object]],
        detection_time: float,
        detections: list[Detection],
    ):
        """Start a new clip for the given camera by appending pre-roll frames from the buffer and initialising a ClipEvent"""
        logger.info("Starting clip with %d pre-roll frames", len(pre_roll_entries))
        start_time = pre_roll_entries[0][0] if pre_roll_entries else detection_time
        clip_path = self._generate_clip_path(camera, start_time)

        Path(clip_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # TODO: Add variable for writer dimensions
        writer = cv2.VideoWriter(clip_path, fourcc, CLIP_OUTPUT_FPS, (1280, 720))

        # TODO: Add retry logic OR consider if its necessary. Keep latency in mind
        if not writer.isOpened():
            logger.error(
                "Failed to open VideoWriter for camera %s at path %s", camera, clip_path
            )
            return

        for _, frame in pre_roll_entries:
            try:
                writer.write(frame)  # Write pre-roll frames to clip
            except Exception:
                logger.exception("Failed to write pre-roll frame for camera %s", camera)

        clip_event = ClipEvent(
            camera=camera,
            clip_path=clip_path,
            writer=writer,
            start_time=start_time,
            last_detection_time=detection_time,
            trigger_detections=[d.to_dict() for d in detections],
        )

        self.active_clips[camera] = clip_event

    def _end_clip(self, camera: str):
        """End the active clip for the given camera by releasing the VideoWriter, saving metadata, and removing the ClipEvent from active_clips"""
        event = self.active_clips.pop(camera, None)

        if event is not None:
            try:
                event.writer.release()
                self.last_clip_end_times[camera] = time.time()
                clip_path, detections = event.clip_path, event.trigger_detections

                start_time = self._format_timestamp(event.start_time)
                end_time = self._format_timestamp(time.time())

                self._save_clip_metadata(clip_path, detections, start_time, end_time)

                logger.info(
                    "Clip saved to %s with %d trigger detections",
                    clip_path,
                    len(detections),
                )
                return
            except Exception:
                logger.exception("Failed to save clip for camera %s", camera)
                return

        logger.info("No active clip to end for camera %s", camera)

    def _save_clip_metadata(
        self,
        clip_path: str,
        detections: list | None,
        start_time: float | None = None,
        end_time: float | None = None,
    ):
        """Save metadata about the clip and its trigger detections to a JSONL file for later analysis"""
        if detections is None:
            detections = []

        metadata_entry = {
            "clip_path": clip_path,
            "detections": detections,
            "start_time": start_time,
            "end_time": end_time,
            "fps": CLIP_OUTPUT_FPS,
            "output_mode": CLIP_OUTPUT_MODE,
        }

        try:
            with open(CLIP_METADATA_PATH, "a") as f:
                json.dump(metadata_entry, f)
                f.write("\n")
        except Exception:
            logger.exception("Failed to save clip metadata for %s", clip_path)

    def _cooldown_over(self, camera: str, now: float) -> bool:
        """Check if cooldown period has passed since last clip for this camera"""
        last_end_time = self.last_clip_end_times.get(camera, 0)
        return now - last_end_time > self.cooldown_seconds

    def _generate_clip_path(self, camera_name: str, timestamp: float) -> str:
        """Generate a unique file path for the clip based on camera name and timestamp"""
        time_str = self._format_timestamp(timestamp)
        filename = f"{camera_name}_{time_str}.mp4"
        return str(CLIP_OUTPUT_DIR / camera_name / filename)

    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp into a string for use in filenames"""
        # TODO: Change timestamp format to ISO. Keep current timestamp for clip file path
        return time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
