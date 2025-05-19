import time
import pyautogui

import mediapipe as mp
import cv2

from action_controller.action_controller import ActionController

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision

video = cv2.VideoCapture(0)

gesture_recognition_options = vision.GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path='gesture_recognition/gesture_recognizer_model/gesture_recognizer.task'
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
pose_landmark_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='tasks/pose_landmarker_heavy.task'),
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False,
    num_poses=1,
)

start = time.perf_counter()

action_controller = ActionController()

# todo: richtiges window fokussieren

with vision.GestureRecognizer.create_from_options(gesture_recognition_options) as gesture_recognizer:
    with vision.PoseLandmarker.create_from_options(pose_landmark_options) as pose_landmark_detection:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp = int((time.perf_counter() - start) * 1000.0)

            gesture_detection_result = gesture_recognizer.recognize_for_video(mp_image, timestamp)
            # gesture_detection_result.hand_landmarks
            # gesture_detection_result.handedness
            # gesture_detection_result.gestures

            action_result = action_controller(gesture_detection_result)
            # action_result.gesture
            # action_result.action (e.g. "point", "prev", "next")
            # action_result.triggered

            pose_result = pose_landmark_detection.detect_for_video(mp_image, timestamp)
            # pose_result.pose_landmarks

            # todo: pointing target detection aufrufen
            # todo: overlay

            # Convert Action to Keypress
            if action_result.action is not None:
                if action_result.triggered:
                    if action_result.action == "prev":
                        pyautogui.press(pyautogui.LEFT)
                    elif action_result.action == "next":
                        pyautogui.press(pyautogui.RIGHT)

            if cv2.waitKey(1) & 0xFF == 27:
                break

video.release()
cv2.destroyAllWindows()
