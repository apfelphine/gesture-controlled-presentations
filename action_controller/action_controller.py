from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Deque

import mediapipe as mp
from mediapipe.tasks.python.components.containers import NormalizedLandmark


class Action(str, Enum):
    NEXT = "next"
    PREV = "prev"
    POINT = "point"


class _Handedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class GestureRecognitionResult:
    gesture: Optional[str] = None
    hand_landmarks: Optional[List[NormalizedLandmark]] = None
    action: Optional[Action] = None


@dataclass
class HandLogGesture:
    last_gestures: Deque[GestureRecognitionResult]
    triggered_action: Optional[Action] = None


@dataclass
class ActionClassificationResult:
    action: Optional[Action] = None
    gesture: Optional[str] = None
    hand: Optional[_Handedness] = None
    triggered: bool = False
    trigger_type: Optional[str] = None  # "frames" or "swipe"
    trigger_value: Optional[float] = None
    trigger_threshold: Optional[float] = None


class ActionController:
    def __init__(self):
        self._enabled_gestures = [
            "pinky-point", "thumb-point", "thumbs-up", "2finger", "point",  # "swipe"
        ]
        self._num_last_hands = 30
        self._min_trigger_frame_count = 4
        self._min_exit_frame_count = 6
        self._min_swipe_distance = 2
        self._last_gesture_recognition_results: Dict[_Handedness, HandLogGesture] = {
            _Handedness.LEFT: HandLogGesture(deque(maxlen=self._num_last_hands)),
            _Handedness.RIGHT: HandLogGesture(deque(maxlen=self._num_last_hands)),
        }

    def __call__(self, gesture_recognizer_result: mp.tasks.vision.GestureRecognizerResult) -> ActionClassificationResult:
        gesture_recognition_result_dict: Dict[_Handedness, GestureRecognitionResult] = {
            _Handedness.LEFT: GestureRecognitionResult(),
            _Handedness.RIGHT: GestureRecognitionResult(),
        }

        if (
            gesture_recognizer_result.gestures and
            gesture_recognizer_result.hand_landmarks and
            gesture_recognizer_result.handedness
        ):
            for gesture_group, hand_landmarks, handedness in zip(
                gesture_recognizer_result.gestures,
                gesture_recognizer_result.hand_landmarks,
                gesture_recognizer_result.handedness,
            ):
                hand_label = handedness[0].category_name.lower()
                if gesture_group:
                    gesture_name = gesture_group[0].category_name
                    if gesture_name in self._enabled_gestures:
                        gesture_recognition_result_dict[hand_label].gesture = gesture_name
                        gesture_recognition_result_dict[hand_label].action = self._get_action_from_gesture(
                            gesture_name, hand_landmarks, hand_label
                        )
                        gesture_recognition_result_dict[hand_label].hand_landmarks = hand_landmarks

        for key, result in gesture_recognition_result_dict.items():
            self._last_gesture_recognition_results[key].last_gestures.append(result)

        result = ActionClassificationResult()
        hand = None
        best_frame = self._num_last_hands

        for key, log in self._last_gesture_recognition_results.items():
            res, frame = self._get_triggered_action_from_last_results(key, list(log.last_gestures))
            res.triggered = res.triggered and res.action != log.triggered_action

            if res.action == log.triggered_action:
                res.trigger_value = res.trigger_threshold

            if res.triggered:
                log.triggered_action = res.action
                if res.action != Action.POINT:
                    log.last_gestures = deque(maxlen=self._num_last_hands)
                else:
                    temp = deque(maxlen=self._num_last_hands)
                    found_first_point = False
                    for r in log.last_gestures:
                        if r.gesture == "point":
                            found_first_point = True
                            temp.append(r)
                        elif found_first_point:
                            temp.append(r)
                    log.last_gestures = temp

            if frame < best_frame or (frame <= best_frame and result.action is None):
                result = res
                best_frame = frame
                hand = key

        for key, log in self._last_gesture_recognition_results.items():
            if key != hand:
                log.triggered_action = None
                log.last_gestures = deque(maxlen=self._num_last_hands)

        return result

    @staticmethod
    def _get_action_from_gesture(
        gesture_name: str,
        hand_landmarks: List[NormalizedLandmark],
        hand: _Handedness,
    ) -> Optional[Action]:
        if gesture_name == "point":
            return Action.POINT

        if gesture_name in ["thumbs-up", "pinky-point"]:
            return Action.NEXT if hand == _Handedness.RIGHT else Action.PREV

        if gesture_name == "thumb-point":
            return Action.PREV if hand == _Handedness.RIGHT else Action.NEXT

        if gesture_name == "2finger":
            pointing_indices = [6, 7, 8, 10, 11, 12]
            pointing_x = [hand_landmarks[i].x for i in pointing_indices]
            non_pointing_x = [l.x for i, l in enumerate(hand_landmarks) if i not in pointing_indices]

            if all(px < npx for px in pointing_x for npx in non_pointing_x):
                return Action.NEXT
            if all(px > npx for px in pointing_x for npx in non_pointing_x):
                return Action.PREV

        return None

    def _get_triggered_action_from_last_results(
        self,
        hand: _Handedness,
        last_results: List[GestureRecognitionResult]
    ) -> (ActionClassificationResult, int):
        gesture_count: Dict[(str, Action), (float, float, int)] = {}
        max_key = None
        max_percent = -1
        last_landmarks = None

        for idx, res in enumerate(reversed(last_results)):
            key = (res.gesture, res.action)
            current, threshold, frame = gesture_count.get(key, (0, 0, self._num_last_hands))
            frame = min(frame, idx)

            if res.gesture == "swipe":
                if last_landmarks is not None and res.hand_landmarks is not None:
                    p1 = res.hand_landmarks[12]
                    p0 = last_landmarks[12]
                    delta_x = p1.x - p0.x
                    avg_z = (abs(p1.z) + abs(p0.z)) / 2 or 0.01
                    scaled_dx = round(delta_x / avg_z, 2)
                    current = current + min(self._min_swipe_distance, scaled_dx) if scaled_dx * current >= 0 else scaled_dx
                    threshold = self._min_swipe_distance
                if threshold != 0:
                    percent = abs(current) / threshold
                else:
                    percent = 0
            else:
                threshold = self._min_trigger_frame_count if res.action else self._min_exit_frame_count
                current = min(threshold, current + 1)
                percent = current / threshold

            gesture_count[key] = current, threshold, frame
            if percent >= 1:
                return ActionClassificationResult(
                    gesture=res.gesture,
                    action=Action.PREV if current < 0 else Action.NEXT if res.gesture == "swipe" else res.action,
                    hand=hand,
                    triggered=True,
                    trigger_type="swipe" if res.gesture == "swipe" else "frames",
                    trigger_value=abs(current),
                    trigger_threshold=threshold
                ), frame

            if percent > max_percent:
                max_percent = percent
                max_key = key

            last_landmarks = res.hand_landmarks

        if max_key:
            current, threshold, frame = gesture_count[max_key]
            gesture, action = max_key
            return ActionClassificationResult(
                gesture=gesture,
                action=Action.PREV if current < 0 else Action.NEXT if gesture == "swipe" else action,
                hand=hand,
                trigger_type="swipe" if gesture == "swipe" else "frames",
                trigger_value=abs(current),
                trigger_threshold=threshold
            ), frame

        return ActionClassificationResult(hand=hand), self._num_last_hands

    def set_enabled_gestures(self, gestures):
        self._enabled_gestures = gestures