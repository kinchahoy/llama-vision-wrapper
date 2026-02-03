"""Minimal camera capture module for 1920x1080 quality images."""

import cv2
from pathlib import Path
from datetime import datetime


class Camera:
    def __init__(self, index=0, width=1920, height=1080):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        """Open camera and configure settings."""
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        # Warmup
        for _ in range(3):
            self.cap.read()
        return self.cap.isOpened()

    def capture(self, filepath=None):
        """Capture a single frame. Returns (success, frame)."""
        if not self.cap or not self.cap.isOpened():
            if not self.open():
                return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        if filepath:
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return True, frame

    def close(self):
        """Release camera."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


def capture_image(output_path=None, width=1920, height=1080, camera_index=0):
    """Quick one-shot capture function."""
    if output_path is None:
        output_path = Path("captures") / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    with Camera(camera_index, width, height) as cam:
        success, _ = cam.capture(output_path)
        return success, output_path if success else None


if __name__ == "__main__":
    success, path = capture_image()
    print(f"{'✓' if success else '✗'} {path}")
