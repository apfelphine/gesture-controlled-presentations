import time
import pyautogui

import mediapipe as mp
import cv2

from action_controller.action_controller import ActionController

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

video = cv2.VideoCapture(0)

gesture_recognition = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path='gesture_recognition/gesture_recognizer_model/gesture_recognizer.task'
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)
start = time.perf_counter()

action_controller = ActionController()

# todo: richtiges window fokussieren

with GestureRecognizer.create_from_options(gesture_recognition) as recognizer:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        gesture_detection_result = recognizer.recognize_for_video(mp_image, int((time.perf_counter() - start) * 1000.0))
        # gesture_detection_result.hand_landmarks
        # gesture_detection_result.handedness
        # gesture_detection_result.gestures

        action_result = action_controller(gesture_detection_result)
        # action_result.gesture
        # action_result.action (e.g. "point", "prev", "next")
        # action_result.triggered

        if action_result.action is not None:
            if action_result.triggered:
                if action_result.action == "prev":
                    pyautogui.press(pyautogui.LEFT)
                elif action_result.action == "next":
                    pyautogui.press(pyautogui.RIGHT)

        # todo: pointing target detection aufrufen
        # todo: overlay

        if cv2.waitKey(1) & 0xFF == 27:
            break

video.release()
cv2.destroyAllWindows()
