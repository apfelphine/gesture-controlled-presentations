import math
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
    min_count: Optional[int] = None
    count: Optional[int] = None
    min_swipe_distance: Optional[float] = None
    swipe_distance: Optional[float] = None


class ActionController:
    def __init__(self):
        self._enabled_gestures = [
            "pinky-point",
            "thumb-point",
            "thumbs-up",
            "2finger",
            #"swipe",
            "point",
        ]
        self._num_last_hands = 30
        self._min_trigger_frame_count = 4
        self._min_exit_frame_count = 6
        self._min_swipe_distance = 2

        self._min_pose_hand_distance = 0.5
        self._min_visibility_pose_detection = 0.5

        self._last_gesture_recognition_results: Dict[_Handedness, HandLogGesture] = {
            _Handedness.LEFT: HandLogGesture(
                last_gestures=deque(maxlen=self._num_last_hands),
                triggered_action=None,
            ),
            _Handedness.RIGHT: HandLogGesture(
                last_gestures=deque(maxlen=self._num_last_hands),
                triggered_action=None,
            ),
        }

    def __call__(
        self,
        gesture_recognizer_result: mp.tasks.vision.GestureRecognizerResult,
        pose_landmarks_result: mp.tasks.vision.PoseLandmarkerResult,
    ) -> ActionClassificationResult:
        gesture_recognition_result: Dict[_Handedness, GestureRecognitionResult] = {
            _Handedness.LEFT: GestureRecognitionResult(),
            _Handedness.RIGHT: GestureRecognitionResult(),
        }

        if (
            gesture_recognizer_result.gestures
            and gesture_recognizer_result.hand_landmarks
            and gesture_recognizer_result.handedness
        ):
            for gesture_group, hand_landmarks, handedness in zip(
                gesture_recognizer_result.gestures,
                gesture_recognizer_result.hand_landmarks,
                gesture_recognizer_result.handedness,
            ):
                hand_label = handedness[0].category_name.lower()

                # Correct handedness using pose landmarks since somtimes the model screws up
                if pose_landmarks_result.pose_landmarks and len(
                    pose_landmarks_result.pose_landmarks[0]
                ):
                    pose_landmarks = pose_landmarks_result.pose_landmarks[0]
                    left_index = 15
                    right_index = 16

                    left_index_landmark = pose_landmarks[left_index]
                    right_index_landmark = pose_landmarks[right_index]

                    if (
                        left_index_landmark
                        and left_index_landmark.visibility
                        <= self._min_visibility_pose_detection
                    ):
                        left_index_landmark = None

                    if (
                        right_index_landmark
                        and right_index_landmark.visibility
                        <= self._min_visibility_pose_detection
                    ):
                        right_index_landmark = None

                    if right_index_landmark and left_index_landmark:
                        x_coords = [l.x for l in hand_landmarks]
                        avg_x = sum(x_coords) / len(x_coords)
                        y_coords = [l.y for l in hand_landmarks]
                        avg_y = sum(y_coords) / len(y_coords)

                        distance_left = round(
                            math.sqrt(
                                math.pow(left_index_landmark.x - avg_x, 2)
                                + math.pow(left_index_landmark.y - avg_y, 2)
                            ),
                            2,
                        )
                        distance_right = round(
                            math.sqrt(
                                math.pow(right_index_landmark.x - avg_x, 2)
                                + math.pow(right_index_landmark.y - avg_y, 2)
                            ),
                            2,
                        )
                        if (
                            distance_left < distance_right
                            and distance_left < self._min_pose_hand_distance
                            and right_index_landmark.visibility
                            < left_index_landmark.visibility
                        ):
                            hand_label = _Handedness.LEFT
                        elif (
                            distance_left > distance_right
                            and distance_right < self._min_pose_hand_distance
                            and right_index_landmark.visibility
                            > left_index_landmark.visibility
                        ):
                            hand_label = _Handedness.RIGHT
                    elif left_index_landmark:
                        hand_label = _Handedness.LEFT
                    elif right_index_landmark:
                        hand_label = _Handedness.RIGHT

                gesture_recognition_result[hand_label].hand_landmarks = hand_landmarks

                if gesture_group:
                    gesture_name = gesture_group[0].category_name
                    if gesture_name and gesture_name in self._enabled_gestures:
                        gesture_recognition_result[hand_label].gesture = gesture_name
                        gesture_recognition_result[hand_label].action = (
                            self._get_action_from_gesture(
                                gesture_name, hand_landmarks, hand_label
                            )
                        )

        for key, value in gesture_recognition_result.items():
            self._last_gesture_recognition_results[key].last_gestures.append(value)

        result = ActionClassificationResult()
        hand = None
        frame = self._num_last_hands

        for key, value in self._last_gesture_recognition_results.items():
            res, f = self._get_triggered_action_from_last_results(
                key, list(value.last_gestures)
            )
            res.triggered = res.triggered and res.action != value.triggered_action

            if res.triggered:
                value.triggered_action = res.action
                if res.action is None or res.action != Action.POINT:
                    value.last_gestures = deque(maxlen=self._num_last_hands)
                if res.action == Action.POINT:
                    temp = deque(maxlen=self._num_last_hands)
                    found_first_point = False
                    for r in list(value.last_gestures):
                        if r.gesture == "point":
                            found_first_point = True
                            temp.append(r)
                        elif found_first_point:
                            temp.append(r)
                    value.last_gestures = temp


            if f < frame or (f <= frame and result.action is None):
                result = res
                frame = f
                hand = key

        for key, value in self._last_gesture_recognition_results.items():
            if key != hand:
                value.triggered_action = None
                value.last_gestures = deque(maxlen=self._num_last_hands)

        return result

    @staticmethod
    def _get_action_from_gesture(
        gesture_name: str, hand_landmarks: List[NormalizedLandmark], hand: _Handedness
    ) -> Optional[Action]:
        if hand_landmarks is None or gesture_name is None:
            return None

        if gesture_name == "point":
            return Action.POINT

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

            pointing_smaller_than_no_pointing = True
            for x in pointing_finger_x:
                if not all(x < x2 for x2 in no_pointing_finger_x):
                    pointing_smaller_than_no_pointing = False
                    break

            if pointing_smaller_than_no_pointing:
                return Action.NEXT

            pointing_bigger_than_no_pointing = True
            for x in pointing_finger_x:
                if not all(x > x2 for x2 in no_pointing_finger_x):
                    pointing_bigger_than_no_pointing = False
                    break

            if pointing_bigger_than_no_pointing:
                return Action.PREV

        return None

    def _get_triggered_action_from_last_results(
        self, hand: _Handedness, last_results: List[GestureRecognitionResult]
    ) -> (ActionClassificationResult, int):
        gesture_count: Dict[(str, Action), (int, int, int)] = {}

        max_percentage_gesture_key = None
        max_percentage = -1

        last_hand_landmarks = None

        for idx, res in enumerate(reversed(last_results)):
            key = (res.gesture, res.action)
            current, _, frame = gesture_count.get(key, (0, 0, self._num_last_hands))

            if idx < frame:
                frame = idx

            if res.gesture == "swipe":
                middle_finger_point_idx = 12
                if last_hand_landmarks:
                    last_point = last_hand_landmarks[middle_finger_point_idx]
                    current_point = res.hand_landmarks[middle_finger_point_idx]

                    delta_x = current_point.x - last_point.x
                    avg_z = (abs(current_point.z) + abs(last_point.z)) / 2
                    depth_scale = max(avg_z, 0.01)
                    scaled_delta_x = delta_x / depth_scale

                    scaled_delta_x = round(scaled_delta_x, 2)

                    if scaled_delta_x * current >= 0:
                        current += scaled_delta_x
                    else:
                        current = scaled_delta_x

                    last_hand_landmarks = res.hand_landmarks

                min_ = self._min_swipe_distance
                percentage = abs(current) / min_
            else:
                current = current + 1
                min_ = (
                    self._min_trigger_frame_count
                    if res.action is not None
                    else self._min_exit_frame_count
                )
                percentage = current / min_

            if res.hand_landmarks:
                last_hand_landmarks = res.hand_landmarks

            gesture_count[key] = current, min_, frame

            if percentage >= 1:
                if res.gesture == "swipe":
                    action = Action.PREV if current < 0 else Action.NEXT
                    return (
                        ActionClassificationResult(
                            gesture=res.gesture,
                            action=action,
                            swipe_distance=current,
                            min_swipe_distance=min_,
                            triggered=True,
                            hand=hand,
                        ),
                        frame,
                    )
                else:
                    return (
                        ActionClassificationResult(
                            gesture=res.gesture,
                            action=res.action,
                            count=current,
                            min_count=min_,
                            triggered=True,
                            hand=hand,
                        ),
                        frame,
                    )

            if percentage > max_percentage:
                max_percentage = percentage
                max_percentage_gesture_key = key

        max_gesture_type = max_percentage_gesture_key[0]

        if max_gesture_type == "swipe":
            action = (
                Action.PREV
                if gesture_count[max_percentage_gesture_key][0] < 0
                else Action.NEXT
            )
            return (
                ActionClassificationResult(
                    gesture=max_percentage_gesture_key[0],
                    action=action,
                    swipe_distance=abs(gesture_count[max_percentage_gesture_key][0]),
                    min_swipe_distance=gesture_count[max_percentage_gesture_key][1],
                    hand=hand,
                ),
                gesture_count[max_percentage_gesture_key][2],
            )
        else:
            return (
                ActionClassificationResult(
                    gesture=max_percentage_gesture_key[0],
                    action=max_percentage_gesture_key[1],
                    count=gesture_count[max_percentage_gesture_key][0],
                    min_count=gesture_count[max_percentage_gesture_key][1],
                    hand=hand,
                ),
                gesture_count[max_percentage_gesture_key][2],
            )
