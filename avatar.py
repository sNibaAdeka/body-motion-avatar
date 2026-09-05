from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_index": 19,
    "right_index": 20,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_foot": 31,
    "right_foot": 32,
}


@dataclass
class AvatarPose:
    points: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def visible(self) -> bool:
        return bool(self.points)


class AvatarMapper:
    """Maps MediaPipe's metric world coordinates to a centred 3D mannequin pose."""

    BONES = (
        ("neck", "nose"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("left_wrist", "left_index"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("right_wrist", "right_index"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_foot"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_foot"),
    )
    GROUND_Y = -4.0

    def __init__(self, smoothing: float, motion_response: float) -> None:
        self.smoothing = smoothing
        self.motion_response = motion_response
        self.previous = AvatarPose()
        self.bone_lengths: dict[tuple[str, str], float] = {}
        self.last_timestamp: float | None = None

    def update(self, world_landmarks: list, timestamp: float) -> AvatarPose:
        if len(world_landmarks) < 33:
            return self.previous
        raw = {
            name: np.array([world_landmarks[index].x, -world_landmarks[index].y, -world_landmarks[index].z], dtype=np.float32)
            for name, index in LANDMARKS.items()
        }
        hip_center = (raw["left_hip"] + raw["right_hip"]) / 2
        shoulder_width = np.linalg.norm(raw["left_shoulder"] - raw["right_shoulder"])
        # Keep the complete figure in view regardless of the user's distance
        # from the camera: shoulder width becomes about 1.1 scene units.
        scale = 1.1 / max(float(shoulder_width), 0.08)
        mapped = {name: (point - hip_center) * scale for name, point in raw.items()}
        mapped["neck"] = (mapped["left_shoulder"] + mapped["right_shoulder"]) / 2
        mapped["pelvis"] = (mapped["left_hip"] + mapped["right_hip"]) / 2
        if self.previous.visible:
            delta_time = max(timestamp - (self.last_timestamp or timestamp), 1 / 120)
            smoothed = {}
            for name, point in mapped.items():
                previous = self.previous.points[name]
                speed = float(np.linalg.norm(point - previous) / delta_time)
                # Fast movement gets a high response (low delay); stillness gets filtering.
                alpha = min(0.90, self.smoothing + speed * self.motion_response)
                smoothed[name] = previous + (point - previous) * alpha
            mapped = smoothed
        self._constrain_bone_lengths(mapped)
        self._place_feet_on_ground(mapped)
        self.previous = AvatarPose(mapped)
        self.last_timestamp = timestamp
        return self.previous

    def _constrain_bone_lengths(self, points: dict[str, np.ndarray]) -> None:
        """Keep each tracked limb physically plausible instead of letting it stretch with noise."""
        for parent, child in self.BONES:
            direction = points[child] - points[parent]
            observed = float(np.linalg.norm(direction))
            if observed < 1e-5:
                continue
            key = (parent, child)
            known_length = self.bone_lengths.get(key)
            if known_length is None:
                known_length = observed
            else:
                known_length = known_length * 0.995 + observed * 0.005
            self.bone_lengths[key] = known_length
            points[child] = points[parent] + direction / observed * known_length

    def _place_feet_on_ground(self, points: dict[str, np.ndarray]) -> None:
        """Translate the whole avatar so its lowest tracked foot rests on the floor."""
        lowest_foot = min(points["left_foot"][1], points["right_foot"][1], points["left_ankle"][1], points["right_ankle"][1])
        offset = self.GROUND_Y - lowest_foot
        for point in points.values():
            point[1] += offset
