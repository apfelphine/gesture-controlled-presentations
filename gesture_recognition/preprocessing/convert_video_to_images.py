import cv2
import os

from tqdm import tqdm

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2.img_hash

from gesture_recognition.preprocessing._utils import detect_and_write_hand

VIDEOS_PATH = "../../data/videos"
OUTPUT_BASE_PATH = "../../data/cropped_hands"

HAND_LANDMARK_MIN_CONFIDENCE = 0.6  # Minimum confidence to detect hand landmarks

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="../../tasks/pose_landmarker_heavy.task"
    ),
    output_segmentation_masks=True,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
)
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="../../tasks/hand_landmarker.task"
    ),
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO,
    min_hand_detection_confidence=HAND_LANDMARK_MIN_CONFIDENCE,
)

def get_video_params(path: str):
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return fps, width, height, n_frames


def get_frames(path: str):
    capture = cv2.VideoCapture(path)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    index = 0
    try:
        while True:
            if capture.grab():
                flag, frame = capture.retrieve()
                if not flag:
                    return
                else:
                    yield frame
                    index += 1
                    if index >= n_frames:
                        return
    finally:
        capture.release()


def main():
    for path in os.listdir(VIDEOS_PATH):
        if not path.endswith(".mp4"):
            continue

        video_path = os.path.join(VIDEOS_PATH, path)
        name = path.strip(".mp4")
        output_path = os.path.join(OUTPUT_BASE_PATH, name)

        fps, width, height, n_frames = get_video_params(video_path)
        print("Video: ", name)
        print("FPS:", fps)
        print("Size:", width, "x", height)
        print("Frames:", n_frames)

        left_hand_output_path = None
        if "LH" in name:
            left_hand_output_path = os.path.join(output_path, "left_hand")
            os.makedirs(left_hand_output_path, exist_ok=True)
        right_hand_output_path = None
        if "RH" in name:
            right_hand_output_path = os.path.join(output_path, "right_hand")
            os.makedirs(right_hand_output_path, exist_ok=True)

        index = 0
        with tqdm(total=n_frames) as pbar:
            pbar.set_description_str(name)

            prev_hashes = []
            with vision.PoseLandmarker.create_from_options(pose_options) as pose:
                with vision.HandLandmarker.create_from_options(hand_options) as hands:

                    for frame in get_frames(video_path):
                        if left_hand_output_path:
                            detect_and_write_hand(
                                hands,
                                pose,
                                frame,
                                index,
                                "left",
                                left_hand_output_path,
                                prev_hashes,
                            )

                        if right_hand_output_path:
                            detect_and_write_hand(
                                hands,
                                pose,
                                frame,
                                index,
                                "right",
                                right_hand_output_path,
                                prev_hashes,
                            )

                        pbar.update(1)
                        index += 1

    print("Done.")


main()
