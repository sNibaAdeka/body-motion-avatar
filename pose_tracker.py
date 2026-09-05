from __future__ import annotations

from dataclasses import dataclass, field

import mediapipe as mp
import numpy as np


@dataclass
class PoseFrame:
    image_landmarks: list = field(default_factory=list)
    world_landmarks: list = field(default_factory=list)


class PoseTracker:
    def __init__(self, model_path: str, detection_confidence: float, tracking_confidence: float) -> None:
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_confidence,
            min_pose_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> PoseFrame:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return PoseFrame()
        world = list(result.pose_world_landmarks[0]) if result.pose_world_landmarks else []
        return PoseFrame(list(result.pose_landmarks[0]), world)

    def close(self) -> None:
        self.landmarker.close()
