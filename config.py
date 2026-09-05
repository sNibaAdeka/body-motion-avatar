from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


@dataclass
class AppConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    analysis_width: int = 960
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65
    smoothing: float = 0.24
    motion_response: float = 0.32
    model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "pose_landmarker_full.task")
