import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

video = cv2.VideoCapture(0)

options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path="gesture_recognizer_model/gesture_recognizer.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

HAND_COLORS = {"Left": (0, 0, 255), "Right": (255, 0, 0)}  # Red  # Blue

timestamp = 0

with GestureRecognizer.create_from_options(options) as recognizer:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip for selfie view
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = recognizer.recognize_for_video(mp_image, timestamp)
        timestamp += 1

        height, width, _ = frame.shape

        if result.gestures and result.hand_landmarks and result.handedness:
            for gesture_group, hand_landmarks, handedness in zip(
                result.gestures, result.hand_landmarks, result.handedness
            ):
                if gesture_group:
                    gesture_name = gesture_group[0].category_name

                    hand_label = handedness[0].category_name

                    # flip hand labels since image was flipped before (kinda hacky but works)
                    if hand_label == "Left":
                        hand_label = "Right"
                    elif hand_label == "Right":
                        hand_label = "Left"

                    color = HAND_COLORS.get(hand_label, (0, 255, 0))  # Fallback: Green

                    xs = [lm.x for lm in hand_landmarks]
                    ys = [lm.y for lm in hand_landmarks]
                    x_min = int(min(xs) * width)
                    x_max = int(max(xs) * width)
                    y_min = int(min(ys) * height)
                    y_max = int(max(ys) * height)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                    if not gesture_name:
                        label = hand_label
                    else:
                        label = f"{hand_label}: {gesture_name}"
                    label_y = y_min - 10 if y_min - 10 > 10 else y_min + 20
                    cv2.putText(
                        frame,
                        label,
                        (x_min, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

video.release()
cv2.destroyAllWindows()
