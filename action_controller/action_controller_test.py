import time

import mediapipe as mp
import cv2

import action_controller

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

video = cv2.VideoCapture(0)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='../gesture_recognition/gesture_recognizer_model/gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)
start = time.perf_counter()

slide_counter = 0
color = (0, 0, 255)
last_action = None

action_controller = action_controller.ActionController()

with GestureRecognizer.create_from_options(options) as recognizer:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = recognizer.recognize_for_video(mp_image, int((time.perf_counter() - start) * 1000.0))

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)  # Flip for selfie view

        action_result = action_controller(result)
        action = action_result.action

        if action != last_action:
            color = (0, 0, 255)

        last_action = action

        if action is not None:
            text = f"{action} ({action_result.gesture})"

            if action_result.swipe_distance is not None:
                text += (f" - swipe distance: {str(abs(round(action_result.swipe_distance, 2)))}/"
                         f"{action_result.min_swipe_distance}")
            else:
                text += f"- count: {action_result.count}/{action_result.min_count}"

            if action_result.triggered:
                color = (0, 255, 0)

                if action == "prev":
                    slide_counter -= 1
                elif action == "next":
                    slide_counter += 1

            frame = cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        x, y, w, h = 256, 256, 80, 80
        cv2.rectangle(frame, (x, x), (x + w, y + h), (255, 255, 255), -1)

        slide_counter_text = str(slide_counter)

        if 10 > slide_counter > 0:
            slide_counter_text = "0" + slide_counter_text
        if slide_counter == 0:
            slide_counter_text = " " + slide_counter_text

        frame = cv2.putText(
            frame,
            slide_counter_text,
        (x + int(w / 10), y + int(h / 2) + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
        (0, 0, 0),
            4
        )

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

video.release()
cv2.destroyAllWindows()
