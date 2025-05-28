from enum import Enum, auto
import cv2, time, numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import mediapipe as mp

class PointerState(Enum):
    IDLE = auto()
    CALIBRATING = auto()
    ACTIVE = auto()

class PointerMode(Enum):
    DOT = 'dot'
    SPOTLIGHT = 'spot'

@dataclass
class PointingResult:
    position: Optional[Tuple[int, int]] = None
    prompt: Optional[str] = None

class PointerController:
    CALIB_DWELL_SEC = 3
    SMOOTH_WIN = 5
    HIDE_TIMEOUT_MS = 300      # grace period against flicker

    def __init__(self, proj_w: int, proj_h: int):
        self.state = PointerState.CALIBRATING
        self.mode = PointerMode.DOT
        self._proj_size = (proj_w, proj_h)
        self._corner_targets = [(0,0),
                                (proj_w-1,0),
                                (proj_w-1,proj_h-1),
                                (0,proj_h-1)]
        self._corner_names = ["top-left","top-right","bottom-right","bottom-left"]
        self._corner_samples = [[] for _ in range(4)]
        self._current_corner = 0
        self._dwell_start = None
        self._H = None                       # 3×3 homography (mapping matrix)
        self._pts_buf = deque(maxlen=self.SMOOTH_WIN)
        self._last_good_px = None
        self._last_time_ms = 0

    def __call__(
            self,
            gesture_recognizer_result: mp.tasks.vision.GestureRecognizerResult,
            action_result,
            frame: np.ndarray
    ) -> PointingResult:
        fingertip_px = None
        if action_result.action == "point":
            if gesture_recognizer_result.hand_landmarks:
                tip = gesture_recognizer_result.hand_landmarks[0][mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                fingertip_px = (int(tip.x * w), int(tip.y * h))

        pointer_pos, prompt = self._process_landmarks(fingertip_px)
        return PointingResult(
            position=pointer_pos,
            prompt=prompt
        )

    def _process_landmarks(self, fingertip_px: tuple[int,int]|None) -> tuple[tuple[int,int]|None,str]:
        now = time.perf_counter() * 1000

        # Calibtration mode (to construct homography)
        if self.state == PointerState.CALIBRATING:
            text = f"Point at {self._corner_names[self._current_corner]} corner"
            if fingertip_px:
                if self._dwell_start is None:
                    self._dwell_start = now
                self._corner_samples[self._current_corner].append(np.float32(fingertip_px))
                if (now - self._dwell_start)/1000 >= self.CALIB_DWELL_SEC:
                    self._current_corner += 1
                    self._dwell_start = None
                    if self._current_corner == 4:
                        self._fit_homography()
                        self.state = PointerState.ACTIVE
                        text = "Calibration complete"
            else:
                self._dwell_start = None
            return None, text

        # Inference mode:
        if fingertip_px is not None:
            self._pts_buf.append(np.float32(fingertip_px))
            self._last_time_ms = now

        # if lost the finger but still inside grace time, keep last pt
        if now - self._last_time_ms < self.HIDE_TIMEOUT_MS and self._pts_buf:
            avg_cam_px = np.mean(self._pts_buf, axis=0)
            proj_pt = cv2.perspectiveTransform(avg_cam_px[None,None,:], self._H)[0,0]
            return (int(proj_pt[0]), int(proj_pt[1])), ""
        else:
            self._pts_buf.clear()
            return None, ""

    def toggle_mode(self):
        self.mode = PointerMode.SPOTLIGHT if self.mode==PointerMode.DOT else PointerMode.DOT

    def _fit_homography(self):
        src = np.array([np.mean(samp,0) for samp in self._corner_samples], dtype=np.float32)
        dst = np.array(self._corner_targets, dtype=np.float32)
        self._H,_ = cv2.findHomography(src, dst, cv2.RANSAC)