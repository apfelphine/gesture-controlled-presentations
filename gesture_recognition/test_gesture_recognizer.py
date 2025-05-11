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
    print("=" * 20)


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer_model/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

timestamp = 0
with GestureRecognizer.create_from_options(options) as recognizer:
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
