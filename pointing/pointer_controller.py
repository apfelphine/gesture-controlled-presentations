from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

_mp_hands = mp.solutions.hands
_mp_pose = mp.solutions.pose

class PointerState(Enum):
    IDLE = auto()
    CALIBRATING = auto()
    ACTIVE = auto()


class PointerMode(Enum):
    DOT = "dot"
    SPOTLIGHT = "spot"


@dataclass
class PointingResult:
    position: Optional[Tuple[int, int]] = None
    prompt: Optional[str] = None
    progress: int = 0


class PointerController:

    SMOOTH_WIN = 5
    HIDE_TIMEOUT_MS = 300
    CALIB_MIN_SAMPLES = 30
    CALIB_STD_FRAC = 0.01
    EDGE_GRACE_FRAC = 0.05

    def __init__(self, proj_w: int, proj_h: int, edge_grace_frac: Optional[float] = None):
        self.state: PointerState = PointerState.CALIBRATING
        self.mode: PointerMode = PointerMode.DOT
        self._proj_size = (proj_w, proj_h)

        self._corner_targets = [
            (0, 0),
            (proj_w - 1, 0),
            (proj_w - 1, proj_h - 1),
            (0, proj_h - 1),
        ]
        self._corner_names = ["top-left", "top-right", "bottom-right", "bottom-left"]

        if edge_grace_frac is not None:
            self.EDGE_GRACE_FRAC = edge_grace_frac

        self._calib_idx = 0
        self._calib_samples: deque[Tuple[int, int]] = deque(maxlen=60)
        self._camera_vectors: List[Tuple[int, int]] = []
        self._corner_progress = 0

        self._smooth_buffer: deque[Tuple[int, int]] = deque(maxlen=self.SMOOTH_WIN)
        self._last_pointer_ts = 0.0

        self._H_vec2scr: Optional[np.ndarray] = None

    def __call__(self, pose_result, gesture_result: mp.tasks.vision.GestureRecognizerResult, action_result, frame: np.ndarray) -> PointingResult:
        keypts = self._extract_points(pose_result, gesture_result, action_result, frame)
        if keypts is not None:
            finger_px, shoulder_px = keypts
            dx_dy = (finger_px[0] - shoulder_px[0], finger_px[1] - shoulder_px[1])
        else:
            dx_dy = None

        pointer_pos, prompt = self._process_vector(dx_dy, frame.shape[:2])
        return PointingResult(position=pointer_pos, prompt=prompt, progress=self._corner_progress)

    def toggle_mode(self):
        self.mode = PointerMode.SPOTLIGHT if self.mode == PointerMode.DOT else PointerMode.DOT

    def set_state(self, state: PointerState):
        self.state = state

    def _extract_points(self, pose_result, gesture_result: mp.tasks.vision.GestureRecognizerResult, action_result, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        if action_result.action != "point":
            return None
        if not gesture_result.hand_landmarks:
            return None

        h, w = frame.shape[:2]

        hand_lm_list = gesture_result.hand_landmarks[0]
        tip_lm = hand_lm_list[_mp_hands.HandLandmark.INDEX_FINGER_TIP]
        finger_px = (int(tip_lm.x * w), int(tip_lm.y * h))

        handedness = (
            gesture_result.handedness[0][0].category_name if gesture_result.handedness else "Right"
        ).lower()

        if pose_result is None or not pose_result.pose_landmarks:
            return None
        
        pose_landmarks = pose_result.pose_landmarks[0]
        shoulder_idx = (
            _mp_pose.PoseLandmark.LEFT_SHOULDER.value if handedness == "left" else _mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        )
        sh_lm = pose_landmarks[shoulder_idx]
        shoulder_px = (int(sh_lm.x * w), int(sh_lm.y * h))

        return finger_px, shoulder_px

    def _process_vector(self, dx_dy: Optional[Tuple[int, int]], frame_hw: Tuple[int, int]):
        if self.state == PointerState.CALIBRATING:
            return self._calibration_step(dx_dy, frame_hw)
        if self.state == PointerState.ACTIVE:
            return self._active_step(dx_dy)
        return None, None

    def _calibration_step(self, dx_dy: Optional[Tuple[int, int]], frame_hw: Tuple[int, int]):
        if self._calib_idx >= 4:
            self._corner_progress = 100
            return None, "complete"

        prompt = self._corner_names[self._calib_idx]

        if dx_dy is not None:
            self._calib_samples.append(dx_dy)

        n = len(self._calib_samples)
        self._corner_progress = min(int(n / self.CALIB_MIN_SAMPLES * 100), 100)

        if n >= self.CALIB_MIN_SAMPLES:
            pts = np.array(self._calib_samples, dtype=np.float32)
            mean = pts.mean(axis=0)
            dists = np.linalg.norm(pts - mean, axis=1)
            std = dists.std()
            img_diag = max(frame_hw)
            if std <= self.CALIB_STD_FRAC * img_diag:
                self._camera_vectors.append(tuple(mean.astype(int)))
                self._calib_idx += 1
                self._calib_samples.clear()
                if self._calib_idx == 4:
                    src = np.array(self._camera_vectors, dtype=np.float32)
                    dst = np.array(self._corner_targets, dtype=np.float32)
                    self._H_vec2scr, _ = cv2.findHomography(src, dst, method=0)
                    self.state = PointerState.ACTIVE
                    self._corner_progress = 100
                    return None, "complete"
        return None, prompt

    def _active_step(self, dx_dy: Optional[Tuple[int, int]]):
        now = time.time()
        pointer_pos = None

        if dx_dy is not None and self._H_vec2scr is not None:
            pt_vec = np.array([[dx_dy]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(pt_vec, self._H_vec2scr)[0, 0]
            x, y = mapped.astype(int)

            w, h = self._proj_size
            gx, gy = int(w * self.EDGE_GRACE_FRAC), int(h * self.EDGE_GRACE_FRAC)
            if -gx <= x <= w - 1 + gx and -gy <= y <= h - 1 + gy:
                self._smooth_buffer.append((x, y))
                if len(self._smooth_buffer) == self.SMOOTH_WIN:
                    xs, ys = zip(*self._smooth_buffer)
                    pointer_pos = (int(np.mean(xs)), int(np.mean(ys)))
                else:
                    pointer_pos = (x, y)
                self._last_pointer_ts = now

        if pointer_pos is None and (now - self._last_pointer_ts) * 1000 > self.HIDE_TIMEOUT_MS:
            self._smooth_buffer.clear()
            return None, None

        return pointer_pos, None