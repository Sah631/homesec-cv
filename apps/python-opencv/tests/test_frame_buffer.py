import os
import unittest

os.environ.setdefault("HV_USER", "user")
os.environ.setdefault("HV_PW", "password")
os.environ.setdefault("NVR_IP", "127.0.0.1")
os.environ.setdefault("PORT", "554")

from video.frame_buffer import FrameBuffer


class FrameBufferTest(unittest.TestCase):
    def test_trims_frames_older_than_max_age(self):
        buffer = FrameBuffer(max_age_seconds=2)

        buffer.add_frame("old", timestamp=1.0)
        buffer.add_frame("kept", timestamp=2.0)
        buffer.add_frame("latest", timestamp=3.1)

        self.assertEqual(buffer.get_frames(), ["kept", "latest"])


if __name__ == "__main__":
    unittest.main()
