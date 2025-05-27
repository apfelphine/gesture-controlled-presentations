def _landmarks_to_dict(landmarks):
    return [
        {
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "visibility": p.visibility,
            "presence": p.presence,
        }
        for p in landmarks
    ]


def _action_result_to_dict(action_result):
    return {
        "action": action_result.action,
        "gesture": action_result.gesture,
        "hand": action_result.hand,
        "triggered": action_result.triggered,
        "min_count": action_result.min_count,
        "count": action_result.count,
        "min_swipe_distance": action_result.min_swipe_distance,
        "swipe_distance": action_result.swipe_distance,
    }


def save_landmarks_to_csv(
    csv_writer, timestamp, pose_result, gesture_result, action_result
):
    pose = []
    if pose_result.pose_landmarks is not None and len(pose_result.pose_landmarks) > 0:
        pose = _landmarks_to_dict(pose_result.pose_landmarks[0])

    hand_left = []
    gesture_left = ""
    hand_right = []
    gesture_right = ""

    if (
        gesture_result.hand_landmarks is not None
        and len(gesture_result.hand_landmarks) > 0
    ):
        hand = gesture_result.handedness[0]
        landmarks = _landmarks_to_dict(gesture_result.hand_landmarks[0])
        gesture = ""
        if gesture_result.gestures[0][0]:
            gesture = gesture_result.gestures[0][0].category_name
        if hand == "Left":
            hand_left = landmarks
            gesture_left = gesture
        else:
            hand_right = landmarks
            gesture_right = gesture

        if len(gesture_result.handedness) > 1:
            hand = gesture_result.handedness[1]
            landmarks = _landmarks_to_dict(gesture_result.hand_landmarks[1])
            gesture = ""
            if gesture_result.gestures[1][0]:
                gesture = gesture_result.gestures[1][0].category_name
            if hand == "Left":
                hand_left = landmarks
                gesture_left = gesture
            else:
                hand_right = landmarks
                gesture_right = gesture

    csv_writer.writerow(
        [
            timestamp,
            pose,
            hand_right,
            gesture_right,
            hand_left,
            gesture_left,
            _action_result_to_dict(action_result),
        ]
    )
