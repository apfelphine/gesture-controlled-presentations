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

import cv2.img_hash

VIDEOS_PATH = "../../data/videos"
OUTPUT_BASE_PATH = "../../data/cropped_hands"

FINAL_IMAGE_SIZE = 256

VISIBILITY_THRESHOLD = 0.3  # Minimum visibility in pose to count a hand as detected
POSE_HAND_BUFFER_PERCENTAGE = (
    4.0  # Percentage to extend detected palm rectangle by for hand landmark recognition
)
HAND_LANDMARK_MIN_CONFIDENCE = 0.5  # Minimum confidence to detect hand landmarks
HAND_LANDMARK_BUFFER_PERCENTAGE = (
    0.1  # Percentage to extend detected hand by for final image
)

HASH_DISTANCE_THRESHOLD = 20  # higher -> images need to be more different

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

hasher = cv2.img_hash.PHash_create()
previous_hashes = []


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


def bounding_square(
    points: List[Tuple[float, float]],
    max_width,
    max_height,
    buffer_percent: float = 0.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
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

    min_x = min(max_width - 1, max(0, min_x))
    min_y = min(max_height - 1, max(0, min_y))
    max_x = max(0, min(max_width - 1, max_x))
    max_y = max(0, min(max_height - 1, max_y))

    # Make it square
    width = max_x - min_x
    height = max_y - min_y
    side = max(width, height)

    # Center the square around the original rectangle
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    half_side = side / 2
    square_min_x = center_x - half_side
    if square_min_x < 0:
        center_x += abs(square_min_x)
        square_min_x = 0

    square_max_x = center_x + half_side
    if square_max_x > max_width - 1:
        square_min_x -= abs(max_width - square_max_x - 1)
        square_max_x = max_width - 1

    square_min_y = center_y - half_side
    if square_min_y < 0:
        center_y += abs(square_min_y)
        square_min_y = 0

    square_max_y = center_y + half_side
    if square_max_y > max_height - 1:
        square_max_y -= abs(max_height - square_max_y - 1)
        square_max_y = max_height - 1

    return (int(square_min_x), int(square_min_y)), (
        int(square_max_x),
        int(square_max_y),
    )


def get_hand_rectangle(pose, index, frame, hand: Literal["left", "right"]):
    height, width, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = pose.detect_for_video(mp_image, index)

    indices = [
        PoseLandmark.LEFT_WRIST,
        PoseLandmark.LEFT_PINKY,
        PoseLandmark.LEFT_THUMB,
        PoseLandmark.LEFT_INDEX,
    ]
    if hand == "right":
        indices = [
            PoseLandmark.RIGHT_WRIST,
            PoseLandmark.RIGHT_PINKY,
            PoseLandmark.RIGHT_THUMB,
            PoseLandmark.RIGHT_INDEX,
        ]

    if result.pose_landmarks:
        landmarks = [list(result.pose_landmarks)[0][index] for index in indices]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        visibilities = [lm.visibility for lm in landmarks]
        min_visibility = min(visibilities)
        return (
            bounding_square(points, width, height, POSE_HAND_BUFFER_PERCENTAGE),
            min_visibility,
            result.segmentation_masks[0],
        )

    return ((0, 0), (0, 0)), 0, None


def draw_rectangle(frame, rect, color=(0, 255, 0), thickness=3):
    start, end = rect
    return cv2.rectangle(frame.copy(), start, end, color, thickness)


def draw_cropped(frame, rect):
    start, end = rect
    w = end[0] - start[0]
    h = end[1] - start[1]
    frame = frame.copy()
    return frame[start[1] : start[1] + h, start[0] : start[0] + w]


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
    hand_recogniser,
    pose_recogniser,
    frame,
    index: int,
    hand: Literal["left", "right"],
    out_path,
    prev_hashes,
):
    rect, visibility, segmentation_mask_frame = get_hand_rectangle(
        pose_recogniser, index, frame, hand
    )
    # rect = rectangular area which contains the hand
    # visibility = visibility of least visible hand landmark
    # segmentation_mask_frame = segmentation_mask of person on entire frame

    if visibility >= VISIBILITY_THRESHOLD:
        cropped_palm = draw_cropped(frame, rect)

        if cropped_palm is not None and len(cropped_palm) != 0:
            visualized_mask = (
                np.repeat(
                    segmentation_mask_frame.numpy_view()[:, :, np.newaxis], 3, axis=2
                )
                * 255
            )
            visualized_mask = visualized_mask.astype(np.uint8)
            cropped_mask = draw_cropped(visualized_mask, rect)

            try:
                cropped_palm_resized = resize(
                    cropped_palm, FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE
                )
                cropped_mask_resized = resize(
                    cropped_mask, FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE
                )
            except Exception as e:
                print("Failed to crop rect ", rect)
                return

            res = hand_recogniser.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_palm_resized),
                index,
            )
            if res.hand_landmarks:
                points = [
                    (int(lm.x * FINAL_IMAGE_SIZE), int(lm.y * FINAL_IMAGE_SIZE))
                    for lm in list(res.hand_landmarks)[0]
                ]

                square = bounding_square(
                    points,
                    FINAL_IMAGE_SIZE,
                    FINAL_IMAGE_SIZE,
                    HAND_LANDMARK_BUFFER_PERCENTAGE,
                )
                cropped_hand = draw_cropped(cropped_palm_resized, square)
                cropped_hand_mask = draw_cropped(cropped_mask_resized, square)

                try:
                    final_hand = resize(
                        cropped_hand, FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE
                    )
                    final_mask = resize(
                        cropped_hand_mask, FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE
                    )

                    final_mask_gray = cv2.cvtColor(final_mask, cv2.COLOR_BGR2GRAY)

                    # Binarize mask (assumes white areas are hand, black is background)
                    _, binary_mask = cv2.threshold(
                        final_mask_gray, 127, 255, cv2.THRESH_BINARY
                    )

                    # Apply mask to the hand
                    final_hand_masked = cv2.bitwise_and(
                        final_hand, final_hand, mask=binary_mask
                    )

                    current_hash = hasher.compute(final_hand)
                    current_hash_mask = hasher.compute(final_hand_masked)

                    is_different = True

                    for h in prev_hashes:
                        dist = cv2.norm(current_hash, h, cv2.NORM_HAMMING)
                        if dist < HASH_DISTANCE_THRESHOLD:
                            is_different = False
                            break
                        dist = cv2.norm(current_hash_mask, h, cv2.NORM_HAMMING)
                        if dist < HASH_DISTANCE_THRESHOLD:
                            is_different = False
                            break

                    if is_different:
                        prev_hashes.append(current_hash)
                        prev_hashes.append(current_hash_mask)
                        cv2.imwrite(
                            os.path.join(out_path, str(index) + ".png"), final_hand
                        )
                except Exception:
                    frame_with_points = cropped_palm_resized.copy()
                    for p in points:
                        frame_with_points = cv2.circle(
                            frame_with_points, p, 1, (0, 0, 255), -1
                        )
                    cv2.imshow("Frame with errors", frame_with_points)
                    if cv2.waitKey(1) & 0xFF == 27:
                        exit(0)


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
