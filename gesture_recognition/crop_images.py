import cv2
import os

import mediapipe as mp
import numpy as np

from PIL import Image
from mediapipe.python.solutions.pose import PoseLandmark
from tqdm import tqdm
from typing import List, Tuple, Literal

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEOS_PATH = "../data/videos"
OUTPUT_BASE_PATH = "../data/cropped_hands"

FINAL_IMAGE_SIZE = 256

VISIBILITY_THRESHOLD = 0.2  # Minimum visibility in pose to count a hand as detected
POSE_HAND_BUFFER_PERCENTAGE = 4.0  # Percentage to extend detected palm rectangle by for hand landmark recognition
HAND_LANDMARK_MIN_CONFIDENCE = 0.5  # Minimum confidence to detect hand landmarks
HAND_LANDMARK_BUFFER_PERCENTAGE = 0.5  # Percentage to extend detected hand by for final image

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
    min_hand_detection_confidence=HAND_LANDMARK_MIN_CONFIDENCE
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


def bounding_square(points: List[Tuple[float, float]], max_width, max_height, buffer_percent: float = 0.0) -> Tuple[
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

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(max_width-1, max_x)
    max_y = min(max_height-1, max_y)

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
        PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY,
        PoseLandmark.LEFT_THUMB, PoseLandmark.LEFT_INDEX
    ]
    if hand == "right":
        indices = [
            PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY,
            PoseLandmark.RIGHT_THUMB, PoseLandmark.RIGHT_INDEX
        ]

    if result.pose_landmarks:
        landmarks = [list(result.pose_landmarks)[0][index] for index in indices]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        visibilities = [lm.visibility for lm in landmarks]
        min_visibility = min(visibilities)
        return bounding_square(points, width, height, POSE_HAND_BUFFER_PERCENTAGE), min_visibility
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


def detect_and_write_hand(
    hand_recogniser, pose_recogniser, frame, index: int, hand: Literal["left", "right"], out_video, out_path
):
    rect, visibility = get_hand_rectangle(pose_recogniser, index, frame, hand)

    if visibility >= VISIBILITY_THRESHOLD:
        cropped_palm = draw_cropped(frame, rect)

        if cropped_palm is not None and len(cropped_palm) != 0:
            cropped_palm_resized = resize(cropped_palm, FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE)
            res = hand_recogniser.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_palm_resized),
                index
            )
            if res.hand_landmarks:
                points = [
                    (int(lm.x * FINAL_IMAGE_SIZE), int(lm.y * FINAL_IMAGE_SIZE))
                    for lm in list(res.hand_landmarks)[0]
                ]

                square = bounding_square(points, FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, HAND_LANDMARK_BUFFER_PERCENTAGE)
                cropped_hand = draw_cropped(cropped_palm_resized, square)

                try:
                    final_hand = resize(cropped_hand, FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE)
                    out_video.write(final_hand)
                    cv2.imwrite(
                        os.path.join(out_path, str(index) + ".png"),
                        final_hand
                    )
                except Exception:
                    frame_with_points = cropped_palm_resized.copy()
                    for p in points:
                        frame_with_points = cv2.circle(frame_with_points, p, 1, (0,0,255), -1)
                    cv2.imshow("AAAA", frame_with_points)
                    if cv2.waitKey(1) & 0xFF == 27:
                        exit(0)
                    pass


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

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out_left_hand = None
        left_hand_output_path = None
        if "LH" in name:
            left_hand_output_path = os.path.join(output_path, "left_hand")
            os.makedirs(left_hand_output_path, exist_ok=True)
            out_left_hand = cv2.VideoWriter(os.path.join(output_path, "left_hand.mp4"), fourcc, fps, (256, 256),
                                            isColor=True)
        out_right_hand = None
        right_hand_output_path = None
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
                        if out_left_hand and left_hand_output_path:
                            detect_and_write_hand(
                                hands, pose, frame, index, "left", out_left_hand, left_hand_output_path
                            )

                        if out_right_hand and right_hand_output_path:
                            detect_and_write_hand(
                                hands, pose, frame, index, "right", out_right_hand, right_hand_output_path
                            )

                        pbar.update(1)
                        index += 1

        if out_left_hand:
            out_left_hand.release()
        if out_right_hand:
            out_right_hand.release()

    print("Done.")


main()
