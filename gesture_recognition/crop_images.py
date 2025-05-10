import cv2
import os

import mediapipe as mp
import numpy as np

from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Literal

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# CONFIG
VIDEOS_PATH = "../data/videos"
OUTPUT_BASE_PATH = "../data/cropped_hands"

# CODE
VISIBILITY_THRESHOLD = 0  # Minimum visibility in pose to count a hand as detected
BUFFER_PERCENTAGE = 2.0  # Percentage to extend detected rectangle by

mp_pose_detection = mp.solutions.pose

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='../tasks/pose_landmarker_heavy.task'),
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
)
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='../tasks/hand_landmarker.task'),
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO,
    min_hand_detection_confidence=0.1
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


def bounding_square(points: List[Tuple[float, float]], buffer_percent: float = 0.0) -> Tuple[
    Tuple[float, float], Tuple[float, float]]:
    if not points:
        raise ValueError("Point list is empty")

    x_values, y_values = zip(*points)
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    # Apply buffer before squaring
    width = max_x - min_x
    height = max_y - min_y

    x_buffer = width * buffer_percent
    y_buffer = height * buffer_percent

    min_x -= x_buffer
    max_x += x_buffer
    min_y -= y_buffer
    max_y += y_buffer

    # Make it square
    width = max_x - min_x
    height = max_y - min_y
    side = max(width, height)

    # Center the square around the original rectangle
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    half_side = side / 2
    square_min_x = center_x - half_side
    square_max_x = center_x + half_side
    square_min_y = center_y - half_side
    square_max_y = center_y + half_side

    return (int(square_min_x), int(square_min_y)), (int(square_max_x), int(square_max_y))


def get_hand_rectangle(pose, index, frame, hand: Literal["left", "right"]):
    height, width, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = pose.detect_for_video(mp_image, index)

    indices = [
        mp_pose_detection.PoseLandmark.LEFT_WRIST, mp_pose_detection.PoseLandmark.LEFT_PINKY,
        mp_pose_detection.PoseLandmark.LEFT_THUMB, mp_pose_detection.PoseLandmark.LEFT_INDEX
    ]
    if hand == "right":
        indices = [
            mp_pose_detection.PoseLandmark.RIGHT_WRIST, mp_pose_detection.PoseLandmark.RIGHT_PINKY,
            mp_pose_detection.PoseLandmark.RIGHT_THUMB, mp_pose_detection.PoseLandmark.RIGHT_INDEX
        ]

    if result.pose_landmarks:
        landmarks = [list(result.pose_landmarks)[0][index] for index in indices]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        visibilities = [lm.visibility for lm in landmarks]
        visibility = min(visibilities)
        return bounding_square(points, BUFFER_PERCENTAGE), visibility
    return ((0, 0), (0, 0)), 0


def draw_rectangle(frame, rect, color=(0, 255, 0), thickness=3):
    start, end = rect
    return cv2.rectangle(frame.copy(), start, end, color, thickness)


def draw_cropped(frame, rect):
    start, end = rect
    w = end[0] - start[0]
    h = end[1] - start[1]
    frame = frame.copy()
    return frame[start[1]:start[1] + h, start[0]:start[0] + w]


def resize(img, width, height):
    if len(img) == 0:
        return None
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((width, height))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def main():
    for path in os.listdir(VIDEOS_PATH):
        if not path.endswith(".mp4"):
            continue

        video_path = os.path.join(VIDEOS_PATH, path)
        name = video_path.strip("../Videos\\").strip(".mp4")
        output_path = os.path.join(OUTPUT_BASE_PATH, name)

        fps, width, height, n_frames = get_video_params(video_path)
        print("Video: ", name)
        print("FPS:", fps)
        print("Size:", width, "x", height)
        print("Frames:", n_frames)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # rectangles_output_path = os.path.join(output_path, "rectangles")
        # os.makedirs(rectangles_output_path, exist_ok=True)
        out_rectangle = cv2.VideoWriter(os.path.join(output_path, "rectangles.mp4"), fourcc, fps, (width, height),
                                        isColor=True)

        if "LH" in name:
            left_hand_output_path = os.path.join(output_path, "left_hand")
            os.makedirs(left_hand_output_path, exist_ok=True)
            out_left_hand = cv2.VideoWriter(os.path.join(output_path, "left_hand.mp4"), fourcc, fps, (256, 256),
                                            isColor=True)

        if "RH" in name:
            right_hand_output_path = os.path.join(output_path, "right_hand")
            os.makedirs(right_hand_output_path, exist_ok=True)
            out_right_hand = cv2.VideoWriter(os.path.join(output_path, "right_hand.mp4"), fourcc, fps, (256, 256),
                                             isColor=True)

        index = 0
        with tqdm(total=n_frames) as pbar:
            pbar.set_description_str(name)

            with vision.PoseLandmarker.create_from_options(pose_options) as pose:
                with vision.HandLandmarker.create_from_options(hand_options) as hands:

                    for frame in get_frames(video_path):
                        frame_with_rects = frame

                        if "LH" in name:
                            lh_rect, lh_visibility = get_hand_rectangle(pose, index, frame, "left")
                            if lh_visibility >= VISIBILITY_THRESHOLD:
                                frame_with_rects = draw_rectangle(frame_with_rects, lh_rect, color=(0, 255, 0),
                                                                  thickness=15)
                                lh_cropped = draw_cropped(frame, lh_rect)

                                if lh_cropped is not None and len(lh_cropped) != 0:
                                    lh_cropped_resized = resize(lh_cropped, 256, 256)
                                    res = hands.detect_for_video(
                                        mp.Image(image_format=mp.ImageFormat.SRGB, data=lh_cropped_resized),
                                        index
                                    )
                                    if res.hand_landmarks:
                                        out_left_hand.write(lh_cropped_resized)
                                        cv2.imwrite(os.path.join(left_hand_output_path, str(index) + ".png"), lh_cropped_resized)

                        if "RH" in name:
                            rh_rect, rh_visibility = get_hand_rectangle(pose, index, frame, "right")
                            if rh_visibility >= VISIBILITY_THRESHOLD:
                                frame_with_rects = draw_rectangle(frame_with_rects, rh_rect, color=(255, 255, 255),
                                                                  thickness=15)
                                rh_cropped = draw_cropped(frame, rh_rect)

                                if rh_cropped is not None and len(rh_cropped) != 0:
                                    rh_cropped_resized = resize(rh_cropped, 256, 256)
                                    res = hands.detect_for_video(
                                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rh_cropped_resized)
                                        , index)
                                    if res.hand_landmarks:
                                        out_right_hand.write(rh_cropped_resized)
                                        cv2.imwrite(os.path.join(right_hand_output_path, str(index) + ".png"), rh_cropped_resized)

                        out_rectangle.write(frame_with_rects)
                        # cv2.imwrite(os.path.join(rectangles_output_path, str(index) + ".png"), frame_with_rects)

                        pbar.update(1)
                        index += 1

        out_rectangle.release()
        if "LH" in name:
            out_left_hand.release()
        if "RH" in name:
            out_right_hand.release()

    print("Done.")


main()
