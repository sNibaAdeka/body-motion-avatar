from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Exercise(str, Enum):
    SQUAT = "SQUATS"
    PUSH_UP = "PUSH-UPS"


@dataclass
class WorkoutStatus:
    exercise: Exercise | None = None
    repetitions: int = 0
    feedback: str = "Press 1 for squats or 2 for push-ups"
    form_ok: bool = False
    phase: str = "SELECT"


def _point(landmarks: list, index: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([landmark.x, -landmark.y, -landmark.z], dtype=np.float32)


def _angle(first: np.ndarray, joint: np.ndarray, third: np.ndarray) -> float:
    left, right = first - joint, third - joint
    divisor = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-6)
    return math.degrees(math.acos(float(np.clip(np.dot(left, right) / divisor, -1.0, 1.0))))


class ExerciseCounter:
    """Counts only completed, technically valid repetitions from 3D pose landmarks."""

    def __init__(self, exercise: Exercise) -> None:
        self.exercise = exercise
        self.repetitions = 0
        self.phase = "UP"
        self.last_rep_at = 0.0

    def reset(self) -> None:
        self.repetitions = 0
        self.phase = "UP"
        self.last_rep_at = 0.0

    def update(self, world_landmarks: list) -> WorkoutStatus:
        if len(world_landmarks) < 33:
            return WorkoutStatus(self.exercise, self.repetitions, "Step back: your whole body must be visible", False, self.phase)
        if self.exercise is Exercise.SQUAT:
            return self._squat(world_landmarks)
        return self._push_up(world_landmarks)

    def _squat(self, landmarks: list) -> WorkoutStatus:
        left_hip, right_hip = _point(landmarks, 23), _point(landmarks, 24)
        left_knee, right_knee = _point(landmarks, 25), _point(landmarks, 26)
        left_ankle, right_ankle = _point(landmarks, 27), _point(landmarks, 28)
        left_angle = _angle(left_hip, left_knee, left_ankle)
        right_angle = _angle(right_hip, right_knee, right_ankle)
        knee_angle = (left_angle + right_angle) / 2
        shoulder_width = max(float(np.linalg.norm(_point(landmarks, 11) - _point(landmarks, 12))), 0.08)
        hip_height = ((left_hip[1] + right_hip[1]) / 2 - (left_knee[1] + right_knee[1]) / 2) / shoulder_width
        standing = knee_angle > 158
        low_enough = knee_angle < 108 and hip_height < 0.95
        if low_enough:
            self.phase = "DOWN"
            return WorkoutStatus(self.exercise, self.repetitions, "Good depth — stand up", True, self.phase)
        if self.phase == "DOWN" and standing:
            self._count()
            self.phase = "UP"
            return WorkoutStatus(self.exercise, self.repetitions, "Good squat counted", True, self.phase)
        if knee_angle < 135 and hip_height >= 0.95:
            return WorkoutStatus(self.exercise, self.repetitions, "Go lower: hips closer to knee level", False, self.phase)
        if knee_angle < 135:
            return WorkoutStatus(self.exercise, self.repetitions, "Bend knees a little more", False, self.phase)
        return WorkoutStatus(self.exercise, self.repetitions, "Stand tall, then squat", True, self.phase)

    def _push_up(self, landmarks: list) -> WorkoutStatus:
        left_shoulder, right_shoulder = _point(landmarks, 11), _point(landmarks, 12)
        left_elbow, right_elbow = _point(landmarks, 13), _point(landmarks, 14)
        left_wrist, right_wrist = _point(landmarks, 15), _point(landmarks, 16)
        left_hip, right_hip = _point(landmarks, 23), _point(landmarks, 24)
        left_knee, right_knee = _point(landmarks, 25), _point(landmarks, 26)
        elbow_angle = (_angle(left_shoulder, left_elbow, left_wrist) + _angle(right_shoulder, right_elbow, right_wrist)) / 2
        body_angle = (_angle(left_shoulder, left_hip, left_knee) + _angle(right_shoulder, right_hip, right_knee)) / 2
        body_straight = body_angle > 155
        lowered = elbow_angle < 98
        raised = elbow_angle > 154
        if lowered and body_straight:
            self.phase = "DOWN"
            return WorkoutStatus(self.exercise, self.repetitions, "Good depth — push up", True, self.phase)
        if self.phase == "DOWN" and raised and body_straight:
            self._count()
            self.phase = "UP"
            return WorkoutStatus(self.exercise, self.repetitions, "Good push-up counted", True, self.phase)
        if not body_straight:
            return WorkoutStatus(self.exercise, self.repetitions, "Keep your back and hips straight", False, self.phase)
        if elbow_angle < 135:
            return WorkoutStatus(self.exercise, self.repetitions, "Lower your chest a little more", False, self.phase)
        return WorkoutStatus(self.exercise, self.repetitions, "Ready — lower with a straight body", True, self.phase)

    def _count(self) -> None:
        now = time.monotonic()
        if now - self.last_rep_at >= 0.55:
            self.repetitions += 1
            self.last_rep_at = now
