import datetime
import math
import os
import random
import time
from enum import Enum
import cv2
import csv

import mediapipe as mp

from action_controller.action_controller import ActionController
from overlay.presentation_overlay import OverlayContextManager
from utils import save_landmarks_to_csv
from pointing.pointer_controller import PointerController, PointerState

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision


class RecordingMode(str, Enum):
    NONE = "none"
    ONLY_LANDMARKS = "only_landmarks"
    CAMERA = "camera"


recording_mode = RecordingMode.ONLY_LANDMARKS
video = cv2.VideoCapture(0)

folder_name = ""
if recording_mode is not RecordingMode.NONE:
    folder_name = f"recordings/{datetime.datetime.now().strftime('%Y%d%m_%H_%M')}_metrics"
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

metrics = []


def cleanup():
    video.release()
    if camera_writer:
        camera_writer.release()
    if csv_file:
        csv_file.close()

    csv_file_metrics = open(f"{folder_name}/metrics.csv", mode="w", newline="")
    csv_writer_metrics = csv.writer(csv_file_metrics)
    csv_writer_metrics.writerow([
        "instruction",
        "hand",
        "time",
        "frame_count",
        "distance"
    ])
    csv_writer_metrics.writerows(metrics)
    csv_file_metrics.close()
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
challenge_start = start
challenge_frame_count = 0

action_controller = ActionController()

hands_down_inst = ("Nehm die Hand runter",
                  lambda action_res,
                         hands: "Left" not in hands and "Right" not in hands and action_res.action is None and action_res.gesture is None)
right_hand_inst = ("Hebe die rechte Hand ohne eine Geste zu zeigen",
                   lambda action_res,
                          hands: "Right" in hands and "Left" not in hands and action_res.action is None and action_res.gesture is None)
left_hand_inst = ("Hebe die linke Hand ohne eine Geste zu zeigen",
                  lambda action_res,
                         hands: "Left" in hands and "Right" not in hands and action_res.action is None and action_res.gesture is None)

gesture_challenges = [
    (right_hand_inst, ("Zeige mit dem kleinen Finger nach oben...",
                       lambda action_res,
                              hands: action_res.gesture == "pinky-point" and action_res.action == "next" and action_res.hand == "right"), hands_down_inst),
    (right_hand_inst, ("Zeige mit deinem Daumen nach links...",
                       lambda action_res,
                              hands: action_res.gesture == "thumb-point" and action_res.action == "prev" and action_res.hand == "right"), hands_down_inst),
    (right_hand_inst, ("Zeige mit zwei Fingern nach links...",
                       lambda action_res,
                              hands: action_res.gesture == "2finger" and action_res.action == "prev" and action_res.hand == "right"), hands_down_inst),
    (right_hand_inst, ("Zeige Daumen hoch",
                       lambda action_res,
                              hands: action_res.gesture == "thumbs-up" and action_res.action == "next" and action_res.hand == "right"), hands_down_inst),

    (left_hand_inst, ("Zeige mit dem kleinen Finger nach oben...",
                      lambda action_res,
                             hands: action_res.gesture == "pinky-point" and action_res.action == "prev" and action_res.hand == "left"), hands_down_inst),
    (left_hand_inst, ("Zeige mit dem Daumen nach rechts...",
                      lambda action_res,
                             hands: action_res.gesture == "thumb-point" and action_res.action == "next" and action_res.hand == "left"), hands_down_inst),
    (left_hand_inst, ("Zeige mit zwei Fingern nach rechts...",
                      lambda action_res,
                             hands: action_res.gesture == "2finger" and action_res.action == "next" and action_res.hand == "left"), hands_down_inst),
    (left_hand_inst, ("Zeige Daumen hoch",
                      lambda action_res,
                             hands: action_res.gesture == "thumbs-up" and action_res.action == "prev" and action_res.hand == "left"), hands_down_inst),
]

latin_square = [gesture_challenges[i:] + gesture_challenges[:i] for i in range(len(gesture_challenges))]
random.seed(42)
random.shuffle(latin_square)

num_runs = sum(
    1 for name in os.listdir("recordings")
    if name.endswith('_metrics') and os.path.isdir(os.path.join("recordings", name))
)

latin_square_num = num_runs % len(latin_square)
gesture_challenges = [item for tup in latin_square[random.randint(0, latin_square_num)] for item in tup]

# Watch out, depending on screen size:
pointing_challenges = [(50, 50), (500, 500), (250, 750), (1250, 1250), (750, 1250), (2500, 50), (2500, 1500)]

challenge_timeout = 1000 * 5

waiting_time = 1000 * 2.5
current_wait_start = None

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
                    now = time.perf_counter()
                    timestamp = int((now - start) * 1000.0)

                    gesture_detection_result = gesture_recognizer.recognize_for_video(
                        mp_image, timestamp
                    )
                    pose_result = pose_landmark_detection.detect_for_video(
                        mp_image, timestamp
                    )

                    action_result = action_controller(
                        gesture_detection_result
                    )

                    calibrating = False

                    if current_wait_start is not None:
                        current_wait_time = int((now - current_wait_start) * 1000.0)
                        if current_wait_time > waiting_time:
                            challenge_frame_count = 0
                            current_wait_start = None
                            challenge_start = now
                    elif len(gesture_challenges) > 0:
                        hands = [h[0].category_name for h in gesture_detection_result.handedness]
                        inst, finished_funct = gesture_challenges[0]

                        if finished_funct(action_result, hands):
                            challenge_time = int((now - challenge_start) * 1000.0)
                            gesture_challenges.pop(0)
                            metrics.append([inst, challenge_time, challenge_frame_count, None])
                            current_wait_start = now

                            if " ohne eine Geste zu zeigen" in inst:
                                instruction = "Hand erkannt. Nutze die Hand für die nächste Aufgabe."
                            elif inst == "Nehm die Hand runter":
                                instruction = "Sehr gut. Keine Hand erkannt."
                                current_wait_start -= 1
                            else:
                                instruction = "Sehr gut. Geste erkannt."
                                current_wait_start -= 0.5
                                metrics.append([inst, action_result.hand, challenge_time, challenge_frame_count, None])
                            overlay.update_instruction(instruction, pointing_controller, color=(60, 240, 60))
                        else:
                            challenge_frame_count += 1

                    if current_wait_start is None:
                        if len(gesture_challenges) > 0:
                            instruction = gesture_challenges[0][0]
                            overlay.update_instruction(instruction, pointing_controller)
                        elif len(pointing_challenges) > 0:
                            action_controller.set_enabled_gestures(["point"])
                            if not pointing_controller.state == PointerState.ACTIVE:
                                # need to calibrate first
                                pointing_result = pointing_controller(gesture_detection_result, action_result, frame)
                                overlay.update_instruction(pointing_result.prompt, pointing_controller,
                                                           pointing_result.progress)
                                overlay.update()
                                challenge_start = now
                            else:
                                pointing_result = pointing_controller(gesture_detection_result, action_result, frame)
                                overlay.update_pointer(
                                    pointing_result.position, pointing_controller.mode
                                )
                                overlay.update_instruction("Zeige auf den Punkt", pointing_controller)
                                point_target = pointing_challenges[0]
                                if pointing_result.position is not None:
                                    distance = math.dist(pointing_result.position, point_target)
                                    radius_target = 36
                                    radius_pointer = 12
                                    challenge_time = int((now - challenge_start) * 1000.0)
                                    if distance < (
                                            radius_pointer + radius_target) or challenge_time >= challenge_timeout:
                                        pointing_challenges.pop(0)
                                        metrics.append([point_target, challenge_time, challenge_frame_count,
                                                        max(0, distance - (radius_pointer + radius_target))])
                                        challenge_start = now
                                        challenge_frame_count = 0
                                    else:
                                        challenge_frame_count += 1

                                if len(pointing_challenges) > 0 and pointing_controller.state != PointerState.CALIBRATING:
                                    point_target = pointing_challenges[0]
                                    overlay.update_pointing_target(point_target)

                    if len(gesture_challenges) == 0 and len(pointing_challenges) == 0:
                        overlay.update_instruction("Messung abgeschlossen. Vielen Dank.", pointing_controller)

                    overlay.update_action_text(
                        gesture_detection_result, action_result, show_action=len(gesture_challenges) == 0
                    )
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
