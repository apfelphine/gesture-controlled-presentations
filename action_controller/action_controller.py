from enum import Enum
from typing import Optional, Dict, List

import mediapipe as mp
from mediapipe.tasks.python.components.containers import NormalizedLandmark

GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult


class Action(str, Enum):
    NEXT = "next"
    PREV = "prev"


class _Handedness(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"


class ActionController:
    def __init__(self):
        self._enabled_gestures = ["pinky-point", "thumb-point", "thumbs-up", "2finger"]

        self._gesture_log: Dict[_Handedness, Dict] = {
            _Handedness.LEFT: {
                "last_gesture": None,
                "count": None,
            },
            _Handedness.RIGHT: {
                "last_gesture": None,
                "count": None,
            }
        }

        self._enter_threshold = 4  # -> 8 frames

    def __call__(self, gesture_recognizer_result: GestureRecognizerResult) -> Dict:
        hands = [_Handedness.LEFT, _Handedness.RIGHT]
        if (
                gesture_recognizer_result.gestures
                and gesture_recognizer_result.hand_landmarks
                and gesture_recognizer_result.handedness
        ):
            for gesture_group, hand_landmarks, handedness in zip(
                    gesture_recognizer_result.gestures,
                    gesture_recognizer_result.hand_landmarks,
                    gesture_recognizer_result.handedness
            ):
                if gesture_group:
                    hand_label = handedness[0].category_name
                    gesture_name = gesture_group[0].category_name

                    if gesture_name and gesture_name in self._enabled_gestures:
                        if hand_label in hands:
                            hands.remove(hand_label)
                        if self._gesture_log[hand_label]["last_gesture"] == gesture_name:
                            self._gesture_log[hand_label]["count"] += 1
                            action = self._get_action_from_gesture(gesture_name, hand_landmarks, hand_label)

                            return {
                                "action": action,
                                "triggered": self._gesture_log[hand_label]["count"] == self._enter_threshold,
                                "min_count": self._enter_threshold,
                                "count": self._gesture_log[hand_label]["count"]
                            }
                        else:
                            self._gesture_log[hand_label]["last_gesture"] = gesture_name
                            self._gesture_log[hand_label]["count"] = 1

                        # todo: swipe
                        # todo: point

        for hand in hands:
            self._gesture_log[hand]["count"] = 0
            self._gesture_log[hand]["last_gesture"] = None

        return {
            "action": None
        }

    @staticmethod
    def _get_action_from_gesture(gesture_name: str, hand_landmarks: List[NormalizedLandmark], hand: _Handedness) -> Optional[Action]:
        if gesture_name in ["thumbs-up",  "pinky-point"]:
            if hand == _Handedness.RIGHT:
                return Action.NEXT
            elif hand == _Handedness.LEFT:
                return Action.PREV

        if gesture_name == "thumb-point":
            if hand == _Handedness.RIGHT:
                return Action.PREV
            elif hand == _Handedness.LEFT:
                return Action.NEXT

        if gesture_name == "2finger":
            pointing_indices = [8, 7, 6, 12, 11, 10]

            pointing_finger_x = []
            no_pointing_finger_x = []

            for idx in range(0, len(hand_landmarks)):
                if idx in pointing_indices:
                    pointing_finger_x.append(hand_landmarks[idx].x)
                else:
                    no_pointing_finger_x.append(hand_landmarks[idx].x)


            pointing_smaller_than_no_pointing = True # next
            for x in pointing_finger_x:
                if not all(x < x2 for x2 in no_pointing_finger_x):
                    pointing_smaller_than_no_pointing = False
                    break

            if pointing_smaller_than_no_pointing:
                return Action.NEXT

            pointing_bigger_than_no_pointing = True  # prev
            for x in pointing_finger_x:
                if not all(x > x2 for x2 in no_pointing_finger_x):
                    pointing_bigger_than_no_pointing = False
                    break

            if pointing_bigger_than_no_pointing:
                return Action.PREV

        return None
