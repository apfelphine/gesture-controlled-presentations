from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

CALIB_DWELL_SEC = 5  # time to hold finger on corner
SMOOTHING_WINDOW = 5
CIRCLE_RADIUS = 15


def find_fingertip(hand_landmarks, image_width: int, image_height: int):
    idx_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    idx_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    idx_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]

    # in normalized coords
    d1 = np.hypot(idx_tip.x - idx_pip.x, idx_tip.y - idx_pip.y)
    d2 = np.hypot(idx_pip.x - idx_mcp.x, idx_pip.y - idx_mcp.y)
    if d1 < d2:
        return None  # finger is not extended

    return int(idx_tip.x * image_width), int(idx_tip.y * image_height)


def exponential_moving_average(prev, new, alpha=0.3):
    if prev is None:
        return new
    return (1 - alpha) * prev + alpha * new


def main():
    parser = argparse.ArgumentParser(description="Pointer demo")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=1280, help="width")
    parser.add_argument("--height", type=int, default=800, help="height")
    args = parser.parse_args()

    proj_size = (args.width, args.height)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    homography = None

    # Buffers for calibration
    corner_names = ["top‑left", "top‑right", "bottom‑right", "bottom‑left"]
    corner_targets = [
        (0, 0),
        (args.width - 1, 0),
        (args.width - 1, args.height - 1),
        (0, args.height - 1),
    ]

    corner_samples: list[list[np.ndarray]] = [[] for _ in range(4)]
    current_corner = 0
    dwell_start = None

    smooth_pt = None
    pts_deque = deque(maxlen=SMOOTHING_WINDOW)

    window_name = "Pointer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *proj_size)

    print("Press c to recalibrate, q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        fingertip_px = None
        if results.multi_hand_landmarks:
            fingertip_px = find_fingertip(results.multi_hand_landmarks[0], cam_w, cam_h)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            # reset calibration
            homography = None
            corner_samples = [[] for _ in range(4)]
            current_corner = 0
            dwell_start = None
            print("Calibration restarted.")

        # Calibration:
        if homography is None:
            blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                f"Point at {corner_names[current_corner]} corner",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            if fingertip_px is not None:
                cv2.circle(frame, fingertip_px, 8, (0, 0, 255), -1)

                if dwell_start is None:
                    dwell_start = time.time()
                elapsed = time.time() - dwell_start
                corner_samples[current_corner].append(
                    np.array(fingertip_px, dtype=np.float32)
                )

                cv2.putText(
                    blank,
                    f"Hold... {elapsed:.1f}s",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                if elapsed >= CALIB_DWELL_SEC:
                    current_corner += 1
                    dwell_start = None
                    print(f"Corner {current_corner} captured.")

                    if current_corner == 4:
                        # Fit homography
                        src_pts = np.array(
                            [np.mean(samples, axis=0) for samples in corner_samples],
                            dtype=np.float32,
                        )
                        dst_pts = np.array(corner_targets, dtype=np.float32)
                        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                        if homography is not None:
                            print("Calibration complete.")
                        else:
                            print("Homography failed -> restart calibration.")
                            current_corner = 0
                            corner_samples = [[] for _ in range(4)]
            else:
                dwell_start = None  # lost finger
            cv2.imshow(window_name, blank)
            cv2.imshow("Webcam", frame)
            continue

        # Runtime:
        blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        if fingertip_px is not None:
            pts_deque.append(np.array(fingertip_px, dtype=np.float32))
            avg_px = np.mean(pts_deque, axis=0)
            smooth_pt = avg_px
        elif len(pts_deque) == 0:
            smooth_pt = None

        if smooth_pt is not None and homography is not None:
            proj_pt = cv2.perspectiveTransform(
                np.array([[smooth_pt]], dtype=np.float32), homography
            )[0][0]
            x, y = int(proj_pt[0]), int(proj_pt[1])
            if 0 <= x < args.width and 0 <= y < args.height:
                cv2.circle(blank, (x, y), CIRCLE_RADIUS, (0, 0, 255), -1)

        cv2.imshow(window_name, blank)
        cv2.imshow("Webcam", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
