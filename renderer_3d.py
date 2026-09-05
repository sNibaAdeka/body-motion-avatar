from __future__ import annotations

import math

import numpy as np
import pyglet
from pyglet.gl import (
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK,
    GL_LIGHT0, GL_LIGHTING, GL_LINES, GL_MODELVIEW, GL_NORMALIZE, GL_POSITION,
    GL_PROJECTION, GL_QUADS, GL_SMOOTH, GLfloat, glBegin, glClear, glClearColor,
    glColor3f, glDisable, glEnable, glEnd, glLightfv, glLineWidth, glLoadIdentity, glMatrixMode,
    glNormal3f, glOrtho, glPolygonMode, glPopMatrix, glPushMatrix, glRotatef, glScalef,
    glShadeModel, glTranslatef, glVertex3f, gluCylinder, gluLookAt, gluNewQuadric,
    gluPerspective, gluSphere, glViewport,
)
from pyglet.window import key

from avatar import AvatarPose


class AvatarWindow(pyglet.window.Window):
    def __init__(self, on_toggle_debug, on_select_exercise, on_reset_workout, on_exit) -> None:
        super().__init__(1280, 720, "Body Motion Avatar", resizable=True, vsync=False)
        self.pose = AvatarPose()
        self.on_toggle_debug_callback = on_toggle_debug
        self.on_select_exercise_callback = on_select_exercise
        self.on_reset_workout_callback = on_reset_workout
        self.on_exit_callback = on_exit
        self.yaw = 0.0
        self.pitch = 0.0
        self.camera_image = None
        self.quadric = gluNewQuadric()
        glClearColor(0.045, 0.06, 0.10, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        light_position = (GLfloat * 4)(2.0, 5.0, 6.0, 1.0)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    def set_pose(self, pose: AvatarPose) -> None:
        self.pose = pose

    def set_camera_frame(self, rgb_frame) -> None:
        height, width = rgb_frame.shape[:2]
        self.camera_image = pyglet.image.ImageData(width, height, "RGB", rgb_frame.tobytes(), pitch=-width * 3)

    def on_key_press(self, symbol, modifiers) -> None:
        if symbol in (key.Q, key.ESCAPE):
            self.on_exit_callback()
        elif symbol == key.D:
            self.on_toggle_debug_callback()
        elif symbol == key._1:
            self.on_select_exercise_callback("squat")
        elif symbol == key._2:
            self.on_select_exercise_callback("push_up")
        elif symbol == key.R:
            self.on_reset_workout_callback()
        elif symbol == key.LEFT:
            self.yaw -= 5
        elif symbol == key.RIGHT:
            self.yaw += 5

    def on_close(self) -> None:
        self.on_exit_callback()

    def on_draw(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # macOS Retina windows have a 2× OpenGL framebuffer. Viewports must use
        # physical framebuffer pixels, otherwise both panes appear tiny/cropped.
        framebuffer_width, framebuffer_height = self.get_framebuffer_size()
        avatar_width = max(framebuffer_width // 2, 1)
        glViewport(0, 0, avatar_width, framebuffer_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(42.0, avatar_width / max(framebuffer_height, 1), 0.1, 100.0)
        target_y, camera_distance = self._camera_frame()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, target_y, camera_distance, 0.0, target_y, 0.0, 0.0, 1.0, 0.0)
        glRotatef(self.pitch, 1, 0, 0)
        glRotatef(self.yaw, 0, 1, 0)
        self._draw_floor()
        if self.pose.visible:
            self._draw_avatar(self.pose)
        self._draw_camera_panel(avatar_width, framebuffer_width, framebuffer_height)

    def _camera_frame(self) -> tuple[float, float]:
        """Keep the complete avatar (head to feet) visible as the tracked pose changes."""
        if not self.pose.visible:
            return -1.4, 10.0
        points = list(self.pose.points.values())
        low = min(float(point[1]) for point in points)
        high = max(float(point[1]) for point in points)
        span = max(4.0, high - low + 1.3)
        # With the 42° vertical FOV, this distance leaves a comfortable border.
        distance = max(8.0, span / (2.0 * math.tan(math.radians(21.0))))
        return (low + high) / 2.0, distance

    def _draw_camera_panel(self, avatar_width: int, framebuffer_width: int, framebuffer_height: int) -> None:
        glViewport(avatar_width, 0, framebuffer_width - avatar_width, framebuffer_height)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, framebuffer_width, 0, framebuffer_height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glColor3f(0.06, 0.08, 0.11)
        glBegin(GL_QUADS)
        glVertex3f(avatar_width, 0, 0)
        glVertex3f(framebuffer_width, 0, 0)
        glVertex3f(framebuffer_width, framebuffer_height, 0)
        glVertex3f(avatar_width, framebuffer_height, 0)
        glEnd()
        if self.camera_image is not None:
            panel_width = framebuffer_width - avatar_width
            # Cover the full right half without distortion; excess width is
            # clipped by that half's OpenGL viewport.
            scale = max(panel_width / self.camera_image.width, framebuffer_height / self.camera_image.height)
            shown_width = int(self.camera_image.width * scale)
            shown_height = int(self.camera_image.height * scale)
            x = avatar_width + (panel_width - shown_width) // 2
            y = (framebuffer_height - shown_height) // 2
            glPushMatrix()
            glTranslatef(x, y, 0)
            glScalef(scale, scale, 1)
            glColor3f(1.0, 1.0, 1.0)
            self.camera_image.blit(0, 0)
            glPopMatrix()
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, framebuffer_width, framebuffer_height)

    def _draw_floor(self) -> None:
        glDisable(GL_LIGHTING)
        glColor3f(0.12, 0.18, 0.28)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for value in range(-6, 7):
            glVertex3f(value, -4.16, -6)
            glVertex3f(value, -4.16, 6)
            glVertex3f(-6, -4.16, value)
            glVertex3f(6, -4.16, value)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_avatar(self, pose: AvatarPose) -> None:
        points = pose.points
        skin = (0.25, 0.78, 0.95)
        joint = (0.96, 0.78, 0.30)
        bones = (
            ("neck", "nose", 0.18),
            ("left_shoulder", "left_elbow", 0.17), ("left_elbow", "left_wrist", 0.14),
            ("left_wrist", "left_index", 0.09),
            ("right_shoulder", "right_elbow", 0.17), ("right_elbow", "right_wrist", 0.14),
            ("right_wrist", "right_index", 0.09),
            ("left_hip", "left_knee", 0.23), ("left_knee", "left_ankle", 0.18),
            ("left_ankle", "left_foot", 0.12),
            ("right_hip", "right_knee", 0.23), ("right_knee", "right_ankle", 0.18),
            ("right_ankle", "right_foot", 0.12),
        )
        self._draw_torso(points, skin)
        for start, end, radius in bones:
            self._cylinder(points[start], points[end], radius, skin)
        for point in points.values():
            self._sphere(point, 0.13, joint)
        self._sphere(points["nose"], 0.42, skin)

    def _draw_torso(self, points: dict[str, np.ndarray], color: tuple[float, float, float]) -> None:
        left_shoulder, right_shoulder = points["left_shoulder"], points["right_shoulder"]
        left_hip, right_hip = points["left_hip"], points["right_hip"]
        glColor3f(*color)
        glBegin(GL_QUADS)
        for depth in (-0.18, 0.18):
            glNormal3f(0, 0, 1 if depth > 0 else -1)
            glVertex3f(*(left_shoulder + np.array([0, 0, depth])))
            glVertex3f(*(right_shoulder + np.array([0, 0, depth])))
            glVertex3f(*(right_hip + np.array([0, 0, depth])))
            glVertex3f(*(left_hip + np.array([0, 0, depth])))
        glEnd()

    def _sphere(self, point: np.ndarray, radius: float, color: tuple[float, float, float]) -> None:
        glPushMatrix()
        glTranslatef(*point)
        glColor3f(*color)
        gluSphere(self.quadric, radius, 14, 10)
        glPopMatrix()

    def _cylinder(self, start: np.ndarray, end: np.ndarray, radius: float, color: tuple[float, float, float]) -> None:
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 1e-4:
            return
        direction = vector / length
        axis = np.cross(np.array([0.0, 0.0, 1.0]), direction)
        angle = math.degrees(math.acos(float(np.clip(direction[2], -1.0, 1.0))))
        glPushMatrix()
        glTranslatef(*start)
        if np.linalg.norm(axis) > 1e-5:
            glRotatef(angle, *axis)
        elif direction[2] < 0:
            glRotatef(180, 1, 0, 0)
        glColor3f(*color)
        gluCylinder(self.quadric, radius, radius * 0.82, length, 12, 1)
        glPopMatrix()
