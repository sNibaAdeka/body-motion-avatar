from __future__ import annotations

import time

import cv2
import pyglet

from avatar import AvatarMapper
from camera import Camera
from config import AppConfig
from exercise_counter import Exercise, ExerciseCounter, WorkoutStatus
from models import ensure_pose_model
from pose_tracker import PoseTracker
from renderer_3d import AvatarWindow


class MotionAvatarApp:
    POSE_CONNECTIONS = (
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 31), (31, 29),
        (24, 26), (26, 28), (28, 32), (32, 30),
    )

    def __init__(self) -> None:
        self.config = AppConfig()
        self.camera = Camera(self.config.camera_index, self.config.camera_width, self.config.camera_height)
        model_path = ensure_pose_model(self.config.model_path)
        self.tracker = PoseTracker(str(model_path), self.config.min_detection_confidence, self.config.min_tracking_confidence)
        self.mapper = AvatarMapper(self.config.smoothing, self.config.motion_response)
        self.debug = True
        self.closed = False
        self.timestamp_ms = 0
        self.exercise: Exercise | None = None
        self.counter: ExerciseCounter | None = None
        self.workout_status = WorkoutStatus()
        self.window = AvatarWindow(self.toggle_debug, self.select_exercise, self.reset_workout, self.close)

    def run(self) -> None:
        pyglet.clock.schedule_interval(self.update, 1 / 60)
        pyglet.app.run()

    def update(self, _delta: float) -> None:
        if self.closed:
            return
        try:
            # Mirror before tracking: the avatar and the right panel act like a mirror.
            display_frame = cv2.flip(self.camera.read(), 1)
            analysis_frame = display_frame
            if analysis_frame.shape[1] > self.config.analysis_width:
                ratio = self.config.analysis_width / analysis_frame.shape[1]
                analysis_frame = cv2.resize(analysis_frame, (self.config.analysis_width, int(analysis_frame.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
            now = int(time.monotonic() * 1000)
            self.timestamp_ms = max(self.timestamp_ms + 1, now)
            result = self.tracker.detect(cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB), self.timestamp_ms)
            self.window.set_pose(self.mapper.update(result.world_landmarks, time.monotonic()))
            if self.counter is not None:
                self.workout_status = self.counter.update(result.world_landmarks)
            self.window.set_camera_frame(self._render_camera_view(display_frame, result.image_landmarks))
        except RuntimeError as error:
            print(f"Camera/pose error: {error}")
            self.close()

    def _render_camera_view(self, frame, landmarks: list):
        shown = frame.copy()
        if self.debug:
            height, width = frame.shape[:2]
            for first, second in self.POSE_CONNECTIONS:
                if first >= len(landmarks) or second >= len(landmarks):
                    continue
                start, end = landmarks[first], landmarks[second]
                if min(getattr(start, "visibility", 1.0), getattr(end, "visibility", 1.0)) > 0.4:
                    cv2.line(shown, (int(start.x * width), int(start.y * height)), (int(end.x * width), int(end.y * height)), (65, 235, 90), 3, cv2.LINE_AA)
            for landmark in landmarks:
                if getattr(landmark, "visibility", 1.0) > 0.4:
                    cv2.circle(shown, (int(landmark.x * width), int(landmark.y * height)), 5, (45, 185, 255), -1, cv2.LINE_AA)
        self._draw_workout_hud(shown)
        return cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)

    def _draw_workout_hud(self, frame) -> None:
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (18, 18), (width - 18, 255), (12, 18, 28), -1)
        if self.exercise is None:
            lines = (("CHOOSE WORKOUT", (250, 250, 250)), ("1 - SQUATS     2 - PUSH-UPS", (80, 230, 135)))
            for index, (text, color) in enumerate(lines):
                cv2.putText(frame, text, (32, 65 + index * 52), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 3, cv2.LINE_AA)
        else:
            color = (80, 230, 135) if self.workout_status.form_ok else (60, 175, 255)
            cv2.putText(frame, self.workout_status.exercise.value, (32, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 3, cv2.LINE_AA)
            count_text = str(self.workout_status.repetitions)
            count_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_DUPLEX, 4.2, 7)
            count_x = (width - count_size[0]) // 2
            cv2.putText(frame, count_text, (count_x, 190), cv2.FONT_HERSHEY_DUPLEX, 4.2, (35, 35, 255), 7, cv2.LINE_AA)
            cv2.putText(frame, self.workout_status.feedback, (32, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

    def toggle_debug(self) -> None:
        self.debug = not self.debug

    def select_exercise(self, value: str) -> None:
        self.exercise = Exercise.SQUAT if value == "squat" else Exercise.PUSH_UP
        self.counter = ExerciseCounter(self.exercise)
        self.workout_status = WorkoutStatus(self.exercise, feedback="Get into the start position", form_ok=True, phase="UP")

    def reset_workout(self) -> None:
        if self.counter is not None:
            self.counter.reset()
            self.workout_status = WorkoutStatus(self.exercise, feedback="Counter reset", form_ok=True, phase="UP")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        pyglet.clock.unschedule(self.update)
        self.tracker.close()
        self.camera.close()
        cv2.destroyAllWindows()
        pyglet.app.exit()


if __name__ == "__main__":
    MotionAvatarApp().run()
