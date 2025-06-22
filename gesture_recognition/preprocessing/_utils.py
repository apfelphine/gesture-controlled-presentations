import os
from typing import Tuple, List, Literal

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from mediapipe.python.solutions.pose import PoseLandmark

FINAL_IMAGE_SIZE = 256

VISIBILITY_THRESHOLD = 0.5  # Minimum visibility in pose to count a hand as detected
POSE_HAND_BUFFER_PERCENTAGE = (
    5  # Percentage to extend detected palm rectangle by for hand landmark recognition
)
HAND_LANDMARK_BUFFER_PERCENTAGE = (
    0.5  # Percentage to extend detected hand by for final image
)

HASH_DISTANCE_THRESHOLD = 15  # higher -> images need to be more different

hasher = cv2.img_hash.PHash_create()
previous_hashes = []

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

    if index is not None:
        result = pose.detect_for_video(mp_image, index)
    else:
        result = pose.detect(mp_image)


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

    return ((0, 0), (0, 0)), None, None


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
    name = None
):
    rect, visibility, segmentation_mask_frame = get_hand_rectangle(
        pose_recogniser, index, frame, hand
    )
    # rect = rectangular area which contains the hand
    # visibility = visibility of least visible hand landmark
    # segmentation_mask_frame = segmentation_mask of person on entire frame

    if visibility is None or visibility >= VISIBILITY_THRESHOLD:
        if visibility is not None:
            cropped_palm = draw_cropped(frame, rect)
        else:
            cropped_palm = frame

        if cropped_palm is not None and len(cropped_palm) != 0:
            if segmentation_mask_frame is not None:
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
                if segmentation_mask_frame is not None:
                    cropped_mask_resized = resize(
                        cropped_mask, FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE
                    )
            except Exception as e:
                return False

            if index is not None:
                res = hand_recogniser.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_palm_resized),
                    index,
                )
            else:
                res = hand_recogniser.detect(
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
                if segmentation_mask_frame is not None:
                    cropped_hand_mask = draw_cropped(cropped_mask_resized, square)

                try:
                    final_hand = resize(
                        cropped_hand, FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE
                    )
                    if segmentation_mask_frame is not None:
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
                        current_hash_mask = hasher.compute(final_hand_masked)

                    current_hash = hasher.compute(final_hand)

                    is_different = True

                    for h in prev_hashes:
                        dist = cv2.norm(current_hash, h, cv2.NORM_HAMMING)
                        if dist < HASH_DISTANCE_THRESHOLD:
                            is_different = False
                            break
                        if segmentation_mask_frame is not None:
                            dist = cv2.norm(current_hash_mask, h, cv2.NORM_HAMMING)
                            if dist < HASH_DISTANCE_THRESHOLD:
                                is_different = False
                                break

                    if is_different:
                        prev_hashes.append(current_hash)
                        if segmentation_mask_frame is not None:
                            prev_hashes.append(current_hash_mask)
                        if index is not None:
                            cv2.imwrite(
                                os.path.join(out_path, str(index) + ".png"), final_hand
                            )
                        else:
                            cv2.imwrite(
                                os.path.join(out_path, name + ".png"), final_hand
                            )
                        return True
                except Exception:
                    frame_with_points = cropped_palm_resized.copy()
                    for p in points:
                        frame_with_points = cv2.circle(
                            frame_with_points, p, 1, (0, 0, 255), -1
                        )
                    cv2.imshow("Frame with errors", frame_with_points)
                    return False
                    if cv2.waitKey(1) & 0xFF == 27:
                        exit(0)
    return False
