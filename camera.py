from __future__ import annotations

import platform

import cv2


class Camera:
    def __init__(self, index: int, width: int, height: int) -> None:
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                "Camera could not be opened. Grant Camera permission to Terminal/Python in "
                "macOS System Settings → Privacy & Security → Camera."
            )

    def read(self):
        success, frame = self.capture.read()
        if not success or frame is None:
            raise RuntimeError("Could not read a frame from the camera.")
        return frame

    def close(self) -> None:
        self.capture.release()
