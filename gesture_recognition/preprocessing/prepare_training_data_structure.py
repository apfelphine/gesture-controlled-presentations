import os
import shutil

import tqdm

INPUT_BASE_PATH = "../../data/cropped_hands"
TRAINING_DATA_PATH = "../../data/training"

NONE_GESTURE = "no-point"

for path in tqdm.tqdm(os.listdir(INPUT_BASE_PATH)):
    parts = path.split("_")

    person = parts[0]
    hand = parts[1]
    gesture = parts[2]

    hand_long = "left_hand" if hand == "LH" else "right_hand"
    picture_path = os.path.join(INPUT_BASE_PATH, path, hand_long)

    if gesture == NONE_GESTURE:
        dest_path = os.path.join(TRAINING_DATA_PATH, "None")

    else:
        if gesture == "thumb-pinky":
            is_inward = parts[3] == "inward"
            if (hand == "LH" and is_inward) or (hand == "RH" and is_inward):
                dest_path = os.path.join(TRAINING_DATA_PATH, "thumb-point")
            else:
                dest_path = os.path.join(TRAINING_DATA_PATH, "pinky-point")
        else:
            dest_path = os.path.join(TRAINING_DATA_PATH, gesture)

    os.makedirs(dest_path, exist_ok=True)
    for filename in os.listdir(picture_path):
        dest_name = f"{person}_{filename.split('.')[0]}_{hand}"
        if len(parts) > 3:
            dest_name += f"_{parts[3]}"
        dest_name += "." + filename.split(".")[1]
        shutil.copyfile(
            os.path.join(picture_path, filename), os.path.join(dest_path, dest_name)
        )
