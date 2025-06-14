import datetime
import os
import time
from enum import Enum
import cv2
import pyautogui
import csv

import mediapipe as mp

from action_controller.action_controller import ActionController
from pointing.pointer_controller import PointerController, PointerState
from overlay.presentation_overlay import OverlayContextManager
from utils import save_landmarks_to_csv

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision


class RecordingMode(str, Enum):
    NONE = "none"
    ONLY_LANDMARKS = "only_landmarks"
    CAMERA = "camera"


recording_mode = RecordingMode.NONE
video = cv2.VideoCapture(0)

folder_name = ""
if recording_mode is not RecordingMode.NONE:
    folder_name = f"recordings/{datetime.datetime.now().strftime('%Y%d%m_%H_%M')}"
    os.makedirs(folder_name, exist_ok=True)

camera_writer = None
if recording_mode == RecordingMode.CAMERA:
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_camera = cv2.VideoWriter_fourcc(*"mp4v")
    camera_writer = cv2.VideoWriter(
        f"{folder_name}/camera.mp4",
        fourcc_camera,
        video.get(cv2.CAP_PROP_FPS) / 4,
        (frame_width, frame_height),
    )

csv_file = None
csv_writer = None
if recording_mode != RecordingMode.NONE:
    csv_file = open(f"{folder_name}/landmarks.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    col_names = [
        "timestamp",
        "pose",
        "hand_right",
        "gesture_right",
        "hand_left",
        "gesture_left",
        "action_result",
    ]
    csv_writer.writerow(col_names)


def cleanup():
    video.release()
    if camera_writer:
        camera_writer.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()


gesture_recognition_options = vision.GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path="gesture_recognition/gesture_recognizer_model/gesture_recognizer.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
pose_landmark_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="tasks/pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False,
    num_poses=1,
)

start = time.perf_counter()

action_controller = ActionController()

try:
    with OverlayContextManager() as overlay:
        pointing_controller = PointerController(overlay.width(), overlay.height())
        with vision.GestureRecognizer.create_from_options(
            gesture_recognition_options
        ) as gesture_recognizer:
            with vision.PoseLandmarker.create_from_options(
                pose_landmark_options
            ) as pose_landmark_detection:
                while video.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=frame_rgb
                    )
                    timestamp = int((time.perf_counter() - start) * 1000.0)

                    gesture_detection_result = gesture_recognizer.recognize_for_video(
                        mp_image, timestamp
                    )
                    pose_result = pose_landmark_detection.detect_for_video(
                        mp_image, timestamp
                    )

                    action_result = action_controller(
                        gesture_detection_result
                    )

                    pointing_result = pointing_controller(
                        gesture_detection_result, action_result, frame
                    )

                    # Running mode:
                    if not pointing_controller.state == PointerState.CALIBRATING:
                        if action_result.action is not None and action_result.triggered:
                            if action_result.action == "prev":
                                pyautogui.press(pyautogui.LEFT)
                            elif action_result.action == "next":
                                pyautogui.press(pyautogui.RIGHT)

                    overlay.update_pointer(
                        pointing_result.position, pointing_controller.mode
                    )
                    overlay.update_instruction(pointing_result.prompt, pointing_controller, pointing_result.progress)
                    overlay.update_action_text(gesture_detection_result, action_result)
                    overlay.update()

                    # Record camera
                    if camera_writer is not None:
                        camera_writer.write(frame)

                    # Record landmarks
                    if csv_writer is not None:
                        save_landmarks_to_csv(
                            csv_writer,
                            timestamp,
                            pose_result,
                            gesture_detection_result,
                            action_result,
                        )

                    if cv2.waitKey(1) & 0xFF == 27:
                        break
except KeyboardInterrupt:
    print("Programm interrupted by user.")
finally:
    cleanup()
