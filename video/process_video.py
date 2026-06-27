import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CAMERA_URLS,
    CLIP_COOLDOWN_SECONDS,
    CLIP_METADATA_PATH,
    CLIP_OUTPUT_DIR,
    CLIP_OUTPUT_FPS,
    CLIP_OUTPUT_MODE,
    CLIP_POST_ROLL_SECONDS,
    DETECTION_CLASSES,
)
from utils.video import draw_frame, extract_detections, is_detection_ignored
from video.frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)


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
    ):
        self.active_clips: dict[str, ClipEvent] = {}
        self.cooldown_seconds = cooldown_seconds
        self.post_roll_seconds = post_roll_seconds
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

        if now - event.last_detection_time > self.post_roll_seconds:
            self._end_clip(camera)  # Finalise and save clip, remove from active_clips

        return

    def _start_clip(
        self,
        camera: str,
        pre_roll_entries: list[tuple[float, object]],
        detection_time: float,
        detections: list,
    ):
        """Start a new clip for the given camera by appending pre-roll frames from the buffer and initialising a ClipEvent"""
        start_time = pre_roll_entries[0][0] if pre_roll_entries else detection_time
        clip_path = self._generate_clip_path(camera, start_time)

        Path(clip_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(clip_path, fourcc, CLIP_OUTPUT_FPS, (1280, 720))

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
            trigger_detections=detections.copy(),
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
        return time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))


def init_frame_buffers(pre_roll_seconds: float) -> dict[str, FrameBuffer]:
    return {
        "cam1": FrameBuffer(pre_roll_seconds=pre_roll_seconds),
        "cam2": FrameBuffer(pre_roll_seconds=pre_roll_seconds),
        "cam3": FrameBuffer(pre_roll_seconds=pre_roll_seconds),
        "cam4": FrameBuffer(pre_roll_seconds=pre_roll_seconds),
    }


def open_video_streams() -> dict[str, cv2.VideoCapture]:
    caps = {}
    for camera_name, url in CAMERA_URLS.items():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        # TODO: Implement retry logic with backoff to make program more robust to temporary network issues or NVR restarts
        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream for {camera_name}")

        caps[camera_name] = cap
        logger.debug("Opened stream for %s", camera_name)

    return caps


def resize_frames(frames: list, target_dims: tuple = (1280, 720)) -> list:
    return [
        cv2.resize(frame, target_dims, interpolation=cv2.INTER_LINEAR)
        for frame in frames
    ]


def process_videos(
    model: YOLO,
    fps: Optional[float] = 20.0,
    pre_roll_seconds: float = 15.0,
    post_roll_seconds: float = 15.0,
    save_clips: bool = False,
    clip_output_mode: str = "annotated",
) -> None:
    # Video processing loop
    logger.debug(
        "Video processing config: fps=%.1f, pre_roll_seconds=%.1f, post_roll_seconds=%.1f, save_clips=%s, clip_output_mode=%s",
        fps,
        pre_roll_seconds,
        post_roll_seconds,
        save_clips,
        clip_output_mode,
    )

    caps = open_video_streams()
    last_inference_time = 0

    frame_buffers = init_frame_buffers(pre_roll_seconds=pre_roll_seconds)
    clip_manager = ClipManager(post_roll_seconds=post_roll_seconds)

    latest_results = [None, None, None, None]

    # Could probably split this whole block into separate threads for each camera
    while True:
        ret1, frame1 = caps["cam1"].read()
        ret2, frame2 = caps["cam2"].read()
        ret3, frame3 = caps["cam3"].read()
        ret4, frame4 = caps["cam4"].read()

        if not ret1 or not ret2 or not ret3 or not ret4:
            logger.error("Failed to grab frame")
            break

        uniform_dims = (1280, 720)

        frames = resize_frames([frame1, frame2, frame3, frame4], uniform_dims)

        now = time.time()

        inference_interval = (
            1.0 / fps if fps else 0.2
        )  # Default to 5 FPS if no FPS specified

        # TODO: Add dynamic fps control to adjust inference frequency based on processing load and frame rate of incoming video streams
        if now - last_inference_time > inference_interval:
            latest_results = model.predict(
                frames,
                save=False,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                classes=DETECTION_CLASSES,
                verbose=False,
            )
            last_inference_time = now

            # TODO: Move detection processing and clip management to separate threads to ensure real-time performance and prevent frame drops, especially when saving clips which can be I/O intensive
            for camera_idx, result in enumerate(latest_results):
                raw_frame = frames[camera_idx]
                annotated_frame = draw_frame(result, raw_frame)

                camera_name = list(CAMERA_URLS.keys())[camera_idx]

                frame_buffers[camera_name].add_frame(
                    raw_frame if clip_output_mode == "raw" else annotated_frame, now
                )

                detections = extract_detections(result) if result is not None else None
                interesting_detections = [
                    detection
                    for detection in detections
                    if not is_detection_ignored(camera_name, detection)
                ]

                logger.debug(
                    "Camera %s detections: total=%d interesting=%d ignored=%d",
                    camera_name,
                    len(detections),
                    len(interesting_detections),
                    len(detections) - len(interesting_detections),
                )

                # Only run clip manager if save clips is enabled
                if save_clips:
                    clip_manager.process_frame(
                        camera=camera_name,
                        frame=raw_frame
                        if clip_output_mode == "raw"
                        else annotated_frame,  # Write raw frames to clip if output mode is raw, otherwise write annotated frames with detections drawn
                        frame_buffer=frame_buffers[camera_name],
                        detections=interesting_detections,
                    )

                # Update frames with annotated versions for display
                frames[camera_idx] = annotated_frame
        else:
            # Draw old annotations to avoid flickering if camera FPS is higher than inference FPS
            for camera_idx, result in enumerate(latest_results):
                frames[camera_idx] = draw_frame(result, frames[camera_idx])

        top_row = np.hstack([frames[0], frames[1]])
        bottom_row = np.hstack([frames[2], frames[3]])
        grid = np.vstack([top_row, bottom_row])

        cv2.imshow("Camera Grid", grid)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Shutdown requested by user")
            break

    logger.debug("Releasing video captures")
    for cap in caps.values():
        cap.release()

    logger.debug("Destroying OpenCV windows")
    cv2.destroyAllWindows()

    logger.info("Video processing stopped")
