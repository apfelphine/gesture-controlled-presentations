import math

import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

video = cv2.VideoCapture(0)

def get_distance(landmark0, landmark1):
    return math.sqrt(
        math.pow(landmark1.x - landmark0.x, 2)
        + math.pow(landmark1.y - landmark0.y, 2)
        + math.pow(landmark1.z - landmark0.z, 2)
    )

# Create a image segmenter instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    for hand in result.handedness:
        print("Hand:", hand[0].category_name)
    for gesture in result.gestures:
        print("Gesture:", gesture[0].category_name)
    for hand_landmarks in result.hand_world_landmarks:
        finger_landmarks = [
            (hand_landmarks[2], hand_landmarks[4]),
            (hand_landmarks[5], hand_landmarks[8]),
            (hand_landmarks[9], hand_landmarks[12]),
            (hand_landmarks[13], hand_landmarks[16]),
            (hand_landmarks[17], hand_landmarks[20]),
        ]
        finger_distances = [get_distance(lm0, lm1) for lm0, lm1 in finger_landmarks]
        finger_thresholds = [
            0.06,
            0.07,
            0.07,
            0.07,
            0.06,
        ]
        index_distance = finger_distances.pop(1)
        if index_distance >= 0.07:
            ok = True
            for finger_distance, finger_threshold in zip(finger_distances, finger_thresholds):
                if finger_distance >= finger_threshold:
                    ok = False
                    break
    print("=" * 80)


options = GestureRecognizerOptions(
    #base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    base_options=BaseOptions(model_asset_path='gesture_recognizer_model/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


timestamp = 0
with GestureRecognizer.create_from_options(options) as recognizer:
  # The recognizer is initialized. Use it here.
    while video.isOpened():
        # Capture frame-by-frame
        ret, frame = video.read()

        if not ret:
            print("Ignoring empty frame")
            break

        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Send live image data to perform gesture recognition
        # The results are accessible via the `result_callback` provided in
        # the `GestureRecognizerOptions` object.
        # The gesture recognizer must be created with the live stream mode.
        recognizer.recognize_async(mp_image, timestamp)

video.release()
cv2.destroyAllWindows()