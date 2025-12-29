# Hand Gesture Laptop Control System

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPTURE LAYER                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Webcam     │───▶│  Frame       │───▶│  Preprocess  │          │
│  │   (OpenCV)   │    │  Buffer      │    │  (Resize/Flip)│          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DETECTION LAYER                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  MediaPipe   │───▶│  Landmark    │───▶│  Hand State  │          │
│  │  Hands       │    │  Extractor   │    │  Classifier  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Gesture     │───▶│  Smoothing   │───▶│  Debounce    │          │
│  │  Recognizer  │    │  Filter      │    │  Logic       │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ACTION LAYER                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Safety      │───▶│  Action      │───▶│  PyAutoGUI   │          │
│  │  Validator   │    │  Executor    │    │  Interface   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

## Gesture-to-Action Mappings

| Gesture | Description | Action | Safety Level |
|---------|-------------|--------|--------------|
| **Open Palm** | All 5 fingers extended, palm facing camera | Mouse cursor control (index finger tip maps to screen) | Safe |
| **Index Point** | Only index finger extended | Mouse move mode (precise control) | Safe |
| **Pinch** | Thumb + Index tips close together (<30px) | Left click (on pinch) | Safe |
| **Pinch Hold** | Pinch maintained >0.3s | Click and drag | Medium |
| **Two Finger Pinch** | Thumb + Middle finger pinch | Right click | Safe |
| **Victory Scroll** | Index + Middle extended, others closed | Scroll mode (vertical hand movement = scroll) | Safe |
| **Fist** | All fingers closed | Pause/Stop all control | Safe |
| **Thumbs Up** | Only thumb extended upward | Media: Play/Pause | Safe |
| **Thumbs Down** | Only thumb extended downward | Media: Volume Down | Safe |
| **Rock Sign** | Index + Pinky extended | Media: Next Track | Safe |

## Landmark Logic (MediaPipe Hand)

```
MediaPipe Hand Landmarks (21 points):
        
    WRIST (0)
        │
    ┌───┴───┬───────┬───────┬───────┐
    │       │       │       │       │
  THUMB   INDEX   MIDDLE   RING   PINKY
   1-4     5-8     9-12   13-16   17-20
    
Finger Detection Logic:
- Finger EXTENDED if: TIP.y < PIP.y (for fingers 1-4)
- Thumb EXTENDED if: TIP.x > IP.x (right hand) or TIP.x < IP.x (left hand)

Key Landmark Indices:
- WRIST: 0
- THUMB: CMC=1, MCP=2, IP=3, TIP=4
- INDEX: MCP=5, PIP=6, DIP=7, TIP=8
- MIDDLE: MCP=9, PIP=10, DIP=11, TIP=12
- RING: MCP=13, PIP=14, DIP=15, TIP=16
- PINKY: MCP=17, PIP=18, DIP=19, TIP=20
```

## Safety Considerations

1. **Dead Zone**: No actions triggered when hand is near screen edges (prevents accidental window close)
2. **Confirmation Delay**: Destructive gestures require 0.5s hold time
3. **Fist Override**: Closing fist immediately stops all actions
4. **Rate Limiting**: Maximum 30 actions per second
5. **Confidence Threshold**: Only act on detections with >70% confidence
6. **Gesture Stability**: Gesture must be stable for 3+ frames before triggering

## Installation

```bash
pip install opencv-python mediapipe pyautogui numpy
```

## Usage

```bash
python gesture_control.py
```

Press 'q' to quit, 'c' to calibrate, 'p' to pause.

## Performance Optimizations for Low-End Laptops

1. Process every 2nd frame (configurable)
2. Reduce camera resolution to 640x480
3. Use MediaPipe's lite model
4. Limit detection to 1 hand
5. Use threading for non-blocking capture
