from collections import deque
import time
from config import CLIP_PRE_ROLL_SECONDS


class FrameBuffer:
    """Timestamped rolling frame buffer."""

    def __init__(self, pre_roll_seconds: float = CLIP_PRE_ROLL_SECONDS):
        self.pre_roll_seconds = pre_roll_seconds
        self.buffer = deque()

    def add_frame(self, frame, timestamp: float | None = None):
        timestamp = time.time() if timestamp is None else timestamp
        frame_to_store = frame.copy() if hasattr(frame, "copy") else frame

        self.buffer.append((timestamp, frame_to_store))
        self.trim(timestamp)

    def trim(self, now: float | None = None):
        now = time.time() if now is None else now
        oldest_allowed = now - self.pre_roll_seconds

        while self.buffer and self.buffer[0][0] < oldest_allowed:
            self.buffer.popleft()

    def get_frames(self):
        return [frame for _, frame in self.buffer]

    def get_entries(self):
        return list(self.buffer)
