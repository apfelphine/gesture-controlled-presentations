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
        self._enabled_gestures = ["pinky-point", "thumb-point", "thumbs-up", "2finger", "swipe"]

        self._gesture_log: Dict[_Handedness, Dict] = {
            _Handedness.LEFT: {
                "last_gesture": None,
                "count": None,
                "last_hand_landmarks": None,
                "swipe_distance": 0,
                "triggered": False
            },
            _Handedness.RIGHT: {
                "last_gesture": None,
                "count": None,
                "last_hand_landmarks": None,
                "swipe_distance": 0,
                "triggered": False
            }
        }

        self._enter_threshold = 4  # -> 4 frames
        self._min_swipe_distance = 0.15

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
                hand_label = handedness[0].category_name
                if hand_label in hands:
                    hands.remove(hand_label)

                if gesture_group:
                    gesture_name = gesture_group[0].category_name

                    if gesture_name and gesture_name in self._enabled_gestures:

                        if self._gesture_log[hand_label]["last_gesture"] == gesture_name:
                            self._gesture_log[hand_label]["count"] += 1
                        else:
                            self._gesture_log[hand_label]["count"] = 1
                            self._gesture_log[hand_label]["swipe_distance"] = 0
                            self._gesture_log[hand_label]["triggered"] = False

                        action = self._get_action_from_gesture(gesture_name, hand_landmarks, hand_label)
                        self._gesture_log[hand_label]["last_hand_landmarks"] = hand_landmarks
                        self._gesture_log[hand_label]["last_gesture"] = gesture_name

                        threshold_met = self._gesture_log[hand_label]["count"] == self._enter_threshold
                        distance_achieved = self._gesture_log[hand_label]["swipe_distance"]
                        min_distance_achieved = abs(distance_achieved) > self._min_swipe_distance

                        triggered = False
                        if (threshold_met and not gesture_name == "swipe") or min_distance_achieved:
                            triggered = not self._gesture_log[hand_label]["triggered"]
                            self._gesture_log[hand_label]["triggered"] = True

                        return {
                            "action": action,
                            "triggered": triggered,
                            "count": self._gesture_log[hand_label]["count"],
                            "min_count": self._enter_threshold,
                            "swipe_distance": distance_achieved,
                            "min_swipe_distance": self._min_swipe_distance
                        }

                self._gesture_log[hand_label]["last_hand_landmarks"] = hand_landmarks
                self._gesture_log[hand_label]["count"] = 0
                self._gesture_log[hand_label]["last_gesture"] = None
                self._gesture_log[hand_label]["swipe_distance"] = 0
                self._gesture_log[hand_label]["triggered"] = False

        for hand in hands:
            self._gesture_log[hand]["count"] = 0
            self._gesture_log[hand]["last_gesture"] = None
            self._gesture_log[hand]["last_hand_landmarks"] = None
            self._gesture_log[hand]["swipe_distance"] = 0
            self._gesture_log[hand]["triggered"] = False

        return {
            "action": None
        }

    def _get_action_from_gesture(self, gesture_name: str, hand_landmarks: List[NormalizedLandmark],
                                 hand: _Handedness) -> Optional[Action]:
        if gesture_name in ["thumbs-up", "pinky-point"]:
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

            pointing_smaller_than_no_pointing = True  # next
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

        if gesture_name == "swipe":
            middle_finger_point_idx = 12
            if not self._gesture_log[hand]["last_hand_landmarks"]:
                return None
            last_x = self._gesture_log[hand]["last_hand_landmarks"][middle_finger_point_idx].x
            current_x = hand_landmarks[middle_finger_point_idx].x

            diff = round(current_x - last_x, 2)
            current_distance = self._gesture_log[hand].get("swipe_distance", 0)

            if diff*current_distance >= 0:  # the diffs are either both positive or both negative (both ok)
                current_distance += diff
                self._gesture_log[hand]["swipe_distance"] = current_distance
                return Action.PREV if current_distance > 0 else Action.NEXT
            else:
                self._gesture_log[hand]["swipe_distance"] = diff

        return None
