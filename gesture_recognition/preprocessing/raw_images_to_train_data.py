import os

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_recognition.preprocessing._utils import detect_and_write_hand

IMAGES_PATH = "../../data/raw-images"
OUTPUT_BASE_PATH = "../../data/training"

HAND_LANDMARK_MIN_CONFIDENCE = 0.6  # Minimum confidence to detect hand landmarks

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="../../tasks/pose_landmarker_heavy.task"
    ),
    output_segmentation_masks=True,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="../../tasks/hand_landmarker.task"
    ),
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE,
    min_hand_detection_confidence=HAND_LANDMARK_MIN_CONFIDENCE,
)

pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)


def process_image(path, gesture_class, prev_hashes):
    image = cv2.imread(path)
    output_path = os.path.join(OUTPUT_BASE_PATH, gesture_class)
    name = (path.split(IMAGES_PATH)[1]
            .replace("/", "-").replace("\\", "-")
            .replace(gesture_class, "").replace(".jpg", "").replace(" ", "")).strip("-")
    os.makedirs(output_path, exist_ok=True)

    if "LH" in path:
        return detect_and_write_hand(
            hand_detector,
            pose_detector,
            image,
            None,
            "left",
            output_path,
            prev_hashes,
            name
        )
    elif "RH" in path:
        return detect_and_write_hand(
            hand_detector,
            pose_detector,
            image,
            None,
            "right",
            output_path,
            prev_hashes,
            name
        )
    else:
        a = detect_and_write_hand(
            hand_detector,
            pose_detector,
            image,
            None,
            "left",
            output_path,
            prev_hashes,
            name
        )
        b = detect_and_write_hand(
            hand_detector,
            pose_detector,
            image,
            None,
            "right",
            output_path,
            prev_hashes,
            name
        )
        return a or b


def iterate_over_directory(directory):
    has_images = False
    prev_hashes = []
    for entry in os.listdir(directory):
        full_entry_path = os.path.join(directory, entry)
        if os.path.isdir(full_entry_path):
            iterate_over_directory(full_entry_path)
        elif entry.endswith(".jpg"):
            if not has_images:
                print(directory)
            res = process_image(full_entry_path, os.path.split(directory)[-1], prev_hashes)
            print("x" if res else ".", end="")
            has_images = True
    if has_images:
        print("")


iterate_over_directory(IMAGES_PATH)

pose_detector.close()
hand_detector.close()
