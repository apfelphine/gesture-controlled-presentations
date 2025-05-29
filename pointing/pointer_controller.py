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


class PointerController:

    CALIB_MIN_SAMPLES: int = 30      # require at least this many points
    CALIB_STD_THRESH: float = 3.0    # stdev in px
    OUTLIER_Z: float = 3.0
    CALIB_MAX_BUFFER: int = 150

    SMOOTH_WIN: int = 5
    HIDE_TIMEOUT_MS: int = 300 

    def __init__(self, proj_w: int, proj_h: int):
        self.state = PointerState.CALIBRATING
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

        # calibration buffers
        self._corner_samples: List[List[np.ndarray]] = [[] for _ in range(4)]
        self._current_corner: int = 0
        self._H: Optional[np.ndarray] = None  # 3×3 homography

        # runtime buffers
        self._pts_buf: deque = deque(maxlen=self.SMOOTH_WIN)
        self._last_time_ms: float = 0.0

    def __call__(
        self,
        gesture_recognizer_result: mp.tasks.vision.GestureRecognizerResult,
        action_result,
        frame: np.ndarray,
    ) -> PointingResult:

        fingertip_px = self._extract_fingertip_px(
            gesture_recognizer_result, action_result, frame
        )

        pointer_pos, prompt = self._process_landmarks(fingertip_px)
        return PointingResult(position=pointer_pos, prompt=prompt)

    def toggle_mode(self) -> None:
        self.mode = PointerMode.SPOTLIGHT if self.mode == PointerMode.DOT else PointerMode.DOT

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

        tip = gesture_recognizer_result.hand_landmarks[0][
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
        ]
        h, w, _ = frame.shape
        return int(tip.x * w), int(tip.y * h)

    def _process_landmarks(
        self, fingertip_px: Optional[Tuple[int, int]]
    ) -> Tuple[Optional[Tuple[int, int]], str]:
        now_ms = time.perf_counter() * 1000

        if self.state == PointerState.CALIBRATING:
            return self._calibration_step(fingertip_px)

        if fingertip_px is not None:
            self._pts_buf.append(np.float32(fingertip_px))
            self._last_time_ms = now_ms

        if (now_ms - self._last_time_ms) < self.HIDE_TIMEOUT_MS and self._pts_buf:
            cam_avg = np.mean(self._pts_buf, axis=0)
            proj_pt = cv2.perspectiveTransform(cam_avg[None, None, :], self._H)[0, 0]
            return (int(proj_pt[0]), int(proj_pt[1])), ""

        self._pts_buf.clear()
        return None, ""

    def _calibration_step(
        self, fingertip_px: Optional[Tuple[int, int]]
    ) -> Tuple[None, str]:
        buf = self._corner_samples[self._current_corner]
        corner_name = self._corner_names[self._current_corner]
        prompt = f"Point at {corner_name} corner"

        if fingertip_px is not None:
            buf.append(np.float32(fingertip_px))
            if len(buf) > self.CALIB_MAX_BUFFER:
                del buf[0]

        n = len(buf)

        if n >= self.CALIB_MIN_SAMPLES:
            cleaned = self._clean_samples_iter(np.asarray(buf))
            self._corner_samples[self._current_corner] = list(cleaned)

            if self._is_stable(cleaned):
                self._current_corner += 1
                if self._current_corner == 4:
                    self._fit_homography()
                    self.state = PointerState.ACTIVE
                    return None, "Calibration complete"
                else:
                    nxt = self._corner_names[self._current_corner]
                    return None, f"Point at {nxt} corner"

        prompt += f" - collected {n}"
        return None, prompt

    @classmethod
    def _clean_samples_iter(cls, samples: np.ndarray) -> np.ndarray:
        keep_mask = np.ones(len(samples), dtype=bool)
        changed = True
        while changed and keep_mask.sum() >= cls.CALIB_MIN_SAMPLES:
            changed = False
            current = samples[keep_mask]
            mu = current.mean(axis=0)
            d = np.linalg.norm(current - mu, axis=1)
            sigma = d.std() or 1e-6

            soft_mask = d < cls.OUTLIER_Z * sigma
            if soft_mask.sum() < keep_mask.sum():
                keep_mask[keep_mask] = soft_mask
                changed = True
                continue

            if sigma > cls.CALIB_STD_THRESH:
                hard_mask = d < (2.0 * cls.CALIB_STD_THRESH)
                if hard_mask.sum() == keep_mask.sum():
                    break  # cannot shrink further
                keep_mask[keep_mask] = hard_mask
                changed = True

        return samples[keep_mask]

    @classmethod
    def _is_stable(cls, samples: np.ndarray) -> bool:
        if len(samples) < cls.CALIB_MIN_SAMPLES:
            return False
        mu = samples.mean(axis=0)
        d = np.linalg.norm(samples - mu, axis=1)
        return d.std() <= cls.CALIB_STD_THRESH

    def _fit_homography(self) -> None:
        src = np.array([np.mean(samples, axis=0) for samples in self._corner_samples],
                       dtype=np.float32)
        dst = np.array(self._corner_targets, dtype=np.float32)
        self._H, _ = cv2.findHomography(src, dst, cv2.RANSAC)