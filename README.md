# Hand gesture based presentation control

## Setup (for developers)

1. Download the mediapipe tasks for hand landmark recognition and pose landmark recognition 
and place them in a `tasks` directory. The models are available under the following links:
   - https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   - https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

2. Place the videos you want to use for training in a `data/videos`. Make sure the videos are 
named in the following schema: `{person / unique run id}_{hand (LH/RH)}_{gesture}_{direction (inward / outward}`.
In these videos show one gesture consistently per video. Do not stop making the gesture, create a nw video for changing the
hand or the direction. 
The following gestures are supported:
   - thumb-pinky
   - swipe
   - point
   - no-point
   - 2finger
   - thumbs-up
