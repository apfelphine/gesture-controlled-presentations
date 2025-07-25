# Hand gesture based presentation control

This project was developed as part of a university course and explores gesture-based interaction 
as an alternative input method. It allows users to control presentation slides using real-time 
hand gesture recognition via a standard webcam. The system uses computer vision techniques to 
detect and classify gestures, supporting three predefined gesture sets for slide navigation (next, previous) 
as well as a laserpointer.
The goal is to provide an intuitive, hands-free experience for presenters without relying on physical clickers or keyboards.

## Usage instructions
This section explains how to set up and run the gesture-controlled presentation tool. You'll find details on required dependencies, how to install them, and how to launch the application. Instructions are also provided for initial calibration and runtime configuration options.

### Requirements:
The tool is developed in Python using `mediapipe` (image classification on edge),
`PyAutoGUI` (Keyboard control), `keyboard` (Keyboard listener), `PyQt5` (user interface).

To set up the tool, follow the following steps:

- Install python 3.11
- Install virtual environment `requirements_windows.txt`

```shell
virtualenv <env_name>
$ source <env_name>/bin/activate
(<env_name>)$ pip install -r requirements.txt
```

- Download https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_light.task and place it in a `tasks` directory

### Run the tool:

To use the tool: Execute the `main.py` script using python.

You will first be instructed to calibrate the Laserpointer. For this, 
point at the different corners of the screen (TOP-RIGHT -> TOP-LEFT ->
BOTTOM-RIGHT -> BOTTOM-LEFT). Keep your hand as stable as possible. Only point to 
the next corner when it says 'corner completed'.

### Configuration:

You may configure the recording level (CAMERA, ONLY_LANDMARKS, NONE) in line 28 of the `main.py` script. By default, 
no data is recorded.

You can configure the dataset to be used via keyboard while the application is running: 
- `ALT GR + 1` will enable all gestures
- `ALT GR + 2` will enable the Laserpointer and the _**Thumb-Pinky**_ gesture for slide control
- `ALT GR + 3` will enable the Laserpointer and the _**Finger guns**_ gesture for slide control
- `ALT GR + 4` will enable the Laserpointer and the _**Thumbs up**_ gesture for slide control

Different color schemes are supported for the user interfaces. 
You may pick one of five color scheme in line 88 of `overlay/presentation_overlay.py`. 
By default, the 'high contrast' color scheme is configured.

## Training a new gesture classification model

1. Download the mediapipe tasks for hand landmark recognition and pose landmark recognition 
and place them in a `tasks` directory. The models are available under the following links:
   - https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   - https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

2. Place the videos you want to use for training in a `data/videos`. Make sure the videos are 
named in the following schema: `{person / unique run id}_{hand (LH/RH)}_{gesture}_{direction (inward / outward}`.
In these videos show one gesture consistently per video. Do not stop making the gesture, create a new video for changing the
hand or the direction. 
The following gestures are supported:
   - thumb-pinky
   - swipe
   - point
   - no-point
   - 2finger
   - thumbs-up

3. Install virtual environments. There are two requirement freezes given - one for Windows (`requirements_windows.txt`) 
and one for Linux (`requirements_linux.txt`). Be aware that training the custom gesture recognizer will only work on
Linux! In order to install the requirements from the file execute the following commands. The python version used is 3.11.2.
4. Execute the following scripts in order:
   - `preprocessing/convert_video_to_images.py` (Windows or Linux)
   - `preprocessing/prepare_training_data_structure.py` (Windows or Linux)
   - `train_gesture_recognizer.py` (only under Linux!)