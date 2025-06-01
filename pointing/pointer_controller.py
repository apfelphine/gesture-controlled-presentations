from enum import Enum, auto
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import mediapipe as mp
import numpy as np


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
    SMOOTH_WIN: int = 5
    HIDE_TIMEOUT_MS: int = 300
    CALIB_MIN_SAMPLES: int = 30
    CALIB_STD_FRAC: float = 0.01
    EDGE_GRACE_FRAC: float = 0.05

    def __init__(self, proj_w: int, proj_h: int, edge_grace_frac: Optional[float] = None):
        self.state: PointerState = PointerState.CALIBRATING
        self.mode: PointerMode = PointerMode.DOT
        self._proj_size: Tuple[int, int] = (proj_w, proj_h)
        self._corner_targets: List[Tuple[int, int]] = [
            (0, 0),
            (proj_w - 1, 0),
            (proj_w - 1, proj_h - 1),
            (0, proj_h - 1),
        ]
        self._corner_names: List[str] = [
            "top-left",
            "top-right",
            "bottom-right",
            "bottom-left",
        ]
        if edge_grace_frac is not None:
            self.EDGE_GRACE_FRAC = edge_grace_frac
        self._calib_idx: int = 0
        self._calib_samples: deque[Tuple[int, int]] = deque(maxlen=120)
        self._camera_corners: List[Tuple[int, int]] = []
        self._smooth_buffer: deque[Tuple[int, int]] = deque(maxlen=self.SMOOTH_WIN)
        self._last_pointer_ts: float = 0.0
        self._H: Optional[np.ndarray] = None
        self._corner_progress: int = 0

    def __call__(
        self,
        gesture_recognizer_result: mp.tasks.vision.GestureRecognizerResult,
        action_result,
        frame: np.ndarray,
    ) -> PointingResult:
        fingertip_px = self._extract_fingertip_px(gesture_recognizer_result, action_result, frame)
        pointer_pos, prompt = self._process_landmarks(fingertip_px, frame.shape[:2])
        return PointingResult(position=pointer_pos, prompt=prompt, progress=self._corner_progress)

    def toggle_mode(self) -> None:
        self.mode = PointerMode.SPOTLIGHT if self.mode == PointerMode.DOT else PointerMode.DOT

    def set_state(self, state: PointerState) -> None:
        self.state = state

    @staticmethod
    def _extract_fingertip_px(
        gesture_recognizer_result: mp.tasks.vision.GestureRecognizerResult,
        action_result,
        frame: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        if action_result.action != "point":
            return None
        if not gesture_recognizer_result.hand_landmarks:
            return None
        tip = gesture_recognizer_result.hand_landmarks[0][mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        h, w = frame.shape[:2]
        return int(tip.x * w), int(tip.y * h)

    def _process_landmarks(
        self,
        fingertip_px: Optional[Tuple[int, int]],
        frame_hw: Tuple[int, int],
    ) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        if self.state == PointerState.CALIBRATING:
            return self._calibration_step(fingertip_px, frame_hw)
        if self.state == PointerState.ACTIVE:
            return self._active_step(fingertip_px)
        return None, None

    def _calibration_step(
        self,
        fingertip_px: Optional[Tuple[int, int]],
        frame_hw: Tuple[int, int],
    ) -> Tuple[Optional[Tuple[int, int]], str]:
        if self._calib_idx >= 4:
            self._corner_progress = 100
            return None, "complete"
        prompt = self._corner_names[self._calib_idx]
        if fingertip_px is not None:
            self._calib_samples.append(fingertip_px)
        n = len(self._calib_samples)
        sample_ratio = min(n / self.CALIB_MIN_SAMPLES, 1.0)
        progress = int(sample_ratio * 100)
        if n >= 2:
            pts = np.array(self._calib_samples, dtype=np.float32)
            mean = pts.mean(axis=0)
            dists = np.linalg.norm(pts - mean, axis=1)
            max_dim = max(frame_hw)
            thresh = max_dim * self.CALIB_STD_FRAC
            if n >= self.CALIB_MIN_SAMPLES and dists.std() > thresh:
                progress = min(progress, 99)
        self._corner_progress = progress
        if n >= self.CALIB_MIN_SAMPLES:
            pts = np.array(self._calib_samples, dtype=np.float32)
            mean = pts.mean(axis=0)
            dists = np.linalg.norm(pts - mean, axis=1)
            max_dim = max(frame_hw)
            thresh = max_dim * self.CALIB_STD_FRAC
            if dists.std() <= thresh:
                self._camera_corners.append(tuple(map(int, mean)))
                self._calib_idx += 1
                self._calib_samples.clear()
                self._corner_progress = 0
                if self._calib_idx == 4:
                    src = np.array(self._camera_corners, dtype=np.float32)
                    dst = np.array(self._corner_targets, dtype=np.float32)
                    self._H, _ = cv2.findHomography(src, dst, method=0)
                    self.state = PointerState.ACTIVE
                    self._corner_progress = 100
                    return None, "complete"
                prompt = self._corner_names[self._calib_idx]
        return None, prompt

    def _active_step(self, fingertip_px: Optional[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], None]:
        self._corner_progress = 100
        now = time.time()
        pointer_pos: Optional[Tuple[int, int]] = None
        if fingertip_px is not None and self._H is not None:
            pt = np.array([[fingertip_px]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(pt, self._H)[0, 0]
            x, y = mapped.astype(int)
            w, h = self._proj_size
            grace_x = int(w * self.EDGE_GRACE_FRAC)
            grace_y = int(h * self.EDGE_GRACE_FRAC)
            if -grace_x <= x <= w - 1 + grace_x and -grace_y <= y <= h - 1 + grace_y:
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
