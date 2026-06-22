import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


os.environ.setdefault("HV_USER", "user")
os.environ.setdefault("HV_PW", "password")
os.environ.setdefault("NVR_IP", "127.0.0.1")
os.environ.setdefault("PORT", "554")

from video.frame_buffer import FrameBuffer
from video.process_video import (
    ClipEventManager,
    LOCATION_MODE_OUTSIDE_IGNORE_ZONES,
    LOCATION_MODE_WHOLE_FRAME,
    center_inside,
    filter_interesting_detections,
)


class DummyWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True

    def isOpened(self):
        return True


class ClipSavingTest(unittest.TestCase):
    def test_center_inside_uses_bbox_center(self):
        self.assertTrue(center_inside([10, 10, 20, 20], [0, 0, 15, 15]))
        self.assertFalse(center_inside([20, 20, 30, 30], [0, 0, 15, 15]))

    def test_filter_interesting_detections_respects_ignore_zones(self):
        detections = [
            {
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.9,
                "bbox": [10, 10, 20, 20],
            },
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.9,
                "bbox": [10, 10, 20, 20],
            },
            {
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.9,
                "bbox": [100, 100, 120, 120],
            },
        ]
        ignore_zones = {
            "cam1": [
                {
                    "name": "parked_car",
                    "bbox": [0, 0, 50, 50],
                    "class_ids": [2],
                }
            ]
        }

        interesting = filter_interesting_detections(
            "cam1",
            detections,
            LOCATION_MODE_OUTSIDE_IGNORE_ZONES,
            ignore_zones,
        )

        self.assertEqual(
            [detection["class_name"] for detection in interesting], ["person", "car"]
        )
        self.assertEqual(
            filter_interesting_detections(
                "cam1",
                detections,
                LOCATION_MODE_WHOLE_FRAME,
                ignore_zones,
            ),
            detections,
        )

    def test_clip_event_starts_extends_closes_and_respects_cooldown(self):
        writers = []

        def writer_factory(path, frame_shape, output_fps):
            writer = DummyWriter()
            writers.append((path, frame_shape, output_fps, writer))
            return writer

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manager = ClipEventManager(
                output_dir=temp_path / "clips",
                metadata_path=temp_path / "metadata" / "clips.jsonl",
                output_fps=10,
                pre_roll_seconds=2,
                post_roll_seconds=2,
                cooldown_seconds=2,
                writer_factory=writer_factory,
            )
            buffer = FrameBuffer(max_age_seconds=2)
            frame0 = np.zeros((4, 4, 3), dtype=np.uint8)
            frame1 = np.ones((4, 4, 3), dtype=np.uint8)
            frame2 = np.full((4, 4, 3), 2, dtype=np.uint8)
            frame3 = np.full((4, 4, 3), 3, dtype=np.uint8)
            detection = {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.9,
                "bbox": [0, 0, 1, 1],
            }

            buffer.add_frame(frame0, timestamp=0.0)
            buffer.add_frame(frame1, timestamp=1.0)
            manager.process_frame("cam1", 1.0, frame1, buffer, [detection])

            self.assertIn("cam1", manager.active_events)
            self.assertEqual(len(writers), 1)
            self.assertEqual(len(writers[0][3].frames), 2)

            buffer.add_frame(frame2, timestamp=2.5)
            manager.process_frame("cam1", 2.5, frame2, buffer, [detection])
            self.assertIn("cam1", manager.active_events)

            buffer.add_frame(frame3, timestamp=4.6)
            manager.process_frame("cam1", 4.6, frame3, buffer, [])

            self.assertNotIn("cam1", manager.active_events)
            self.assertTrue(writers[0][3].released)
            self.assertEqual(manager.last_closed_times["cam1"], 4.6)

            manager.process_frame("cam1", 5.0, frame3, buffer, [detection])
            self.assertEqual(len(writers), 1)

            manager.process_frame("cam1", 6.7, frame3, buffer, [detection])
            self.assertEqual(len(writers), 2)

            metadata = json.loads(manager.metadata_path.read_text().strip())
            self.assertEqual(metadata["camera"], "cam1")
            self.assertEqual(metadata["trigger_detections"], [detection])


if __name__ == "__main__":
    unittest.main()
