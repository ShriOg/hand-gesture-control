# Hand Gesture Laptop Control System - Complete Guide

## Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the system
python run.py

# Or with options
python gesture_control.py --low-end  # For slower laptops
python gesture_control.py --sensitivity 2.0  # Faster cursor
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 8 GB |
| CPU | Dual-core | Quad-core |
| Camera | 480p | 720p |
| OS | Windows 10/11, macOS, Linux | - |

## Gesture Reference

### Cursor Control

#### Open Palm 🖐️
- **Action**: Move cursor across screen
- **Technique**: Extend all 5 fingers, palm facing camera
- **Mapping**: Index finger tip position → Screen coordinates
- **Best for**: Large cursor movements

#### Index Point ☝️  
- **Action**: Precise cursor positioning
- **Technique**: Extend only index finger, curl others
- **Mapping**: Index tip with higher precision
- **Best for**: Clicking small targets

### Clicking Actions

#### Pinch (Left Click) 👌
- **Action**: Single left click
- **Technique**: Touch thumb tip to index finger tip
- **Timing**: Quick pinch = click, Hold > 0.3s = drag
- **Release**: Releasing pinch ends drag or confirms click

#### Two-Finger Pinch (Right Click) 🤏
- **Action**: Right-click context menu
- **Technique**: Touch thumb tip to middle finger tip
- **Note**: Index finger should NOT be touching thumb

### Scrolling

#### Victory/Peace Sign ✌️
- **Action**: Enter scroll mode
- **Technique**: Extend index + middle fingers, curl others
- **Control**: Move hand up = scroll up, down = scroll down
- **Speed**: Movement distance controls scroll speed

### Media Controls

#### Thumbs Up 👍
- **Action**: Play/Pause media
- **Technique**: Extend thumb upward, curl all fingers
- **Works with**: Spotify, YouTube, VLC, system media

#### Thumbs Down 👎
- **Action**: Volume down
- **Technique**: Extend thumb downward, curl all fingers

#### Rock Sign 🤘
- **Action**: Next track
- **Technique**: Extend index + pinky, curl middle/ring/thumb

### System Control

#### Fist ✊
- **Action**: Pause/Resume gesture control
- **Technique**: Curl all fingers into fist
- **Purpose**: Emergency stop, prevent accidental actions
- **Toggle**: Fist again to resume

## Calibration

### When to Calibrate
- First time using the system
- Different lighting conditions
- Different user (hand size varies)
- After system updates

### Calibration Process
1. Press **C** during operation
2. Follow on-screen prompts:
   - Hold open palm for 2 seconds
   - Make pinch gesture for 2 seconds
   - Make fist for 2 seconds
3. System calculates optimal thresholds
4. Calibration auto-saves

### Manual Calibration

Edit thresholds in code or create a profile:

```python
from utils import GestureProfile

profile = GestureProfile(
    name="my_profile",
    pinch_threshold=0.06,  # Increase if pinch triggers too easily
    movement_sensitivity=1.2,  # Decrease for slower cursor
)
profile.save("my_profile.json")
```

## Performance Optimization

### For Low-End Laptops

```powershell
python gesture_control.py --low-end
```

This enables:
- Process every 2nd frame (saves 50% CPU)
- Lower camera resolution (480x360)
- Reduced detection confidence (faster processing)
- Lite MediaPipe model

### Manual Tuning

```python
config = GestureConfig()
config.process_every_n_frames = 3  # Process every 3rd frame
config.frame_width = 320
config.frame_height = 240
config.detection_confidence = 0.5
```

### Performance Tips

1. **Close other applications** using the camera
2. **Reduce browser tabs** - they consume memory
3. **Disable visual effects** in Windows
4. **Use wired power** - laptops throttle on battery
5. **Ensure good lighting** - reduces processing overhead

## Lighting Conditions

### Optimal Setup
- Diffuse, even lighting from front
- Avoid backlighting (window behind you)
- Light source should illuminate your hand
- Avoid harsh shadows

### Handling Poor Lighting

The system includes adaptive preprocessing:

```python
# In advanced_processing.py
preprocessor = AdaptivePreprocessor()
# Automatically adjusts for:
# - Dark conditions: Brightness boost
# - Overexposure: Brightness reduction
# - Low contrast: CLAHE enhancement
# - Noise: Bilateral filtering
```

### Troubleshooting Lighting Issues

| Problem | Solution |
|---------|----------|
| Hand not detected | Add light source in front |
| Jittery detection | Reduce backlighting |
| Delayed response | Improve contrast (solid background) |
| False positives | Avoid patterned sleeves/backgrounds |

## Safety Features

### Dead Zones
- 10% margin at screen edges
- Prevents accidental window close buttons
- Configurable via `screen_margin` parameter

### Debouncing
- Clicks: 300ms minimum interval
- Scroll: 50ms minimum interval
- Media: 500ms minimum interval
- Prevents accidental double-clicks

### Gesture Stability
- Gesture must be stable for 3 frames
- Prevents flicker between gestures
- Kalman filtering for smooth transitions

### Emergency Stop
- **Fist gesture** immediately stops all actions
- PyAutoGUI failsafe enabled at screen corners
- Rate limiting: Max 30 actions/second

### Action Confirmation
- Drag requires 0.3s hold time
- Prevents accidental drag operations
- Visual feedback in UI

## Troubleshooting

### Camera Issues

```powershell
# Check if camera is accessible
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**Camera not found:**
- Check camera permissions in Windows Settings
- Close other apps using camera
- Try different camera index: `--camera 1`

### Detection Issues

**Hand not detected:**
- Ensure hand is fully in frame
- Check lighting conditions
- Try moving closer to camera
- Use solid background

**Wrong gestures recognized:**
- Run calibration (press C)
- Adjust `pinch_threshold` higher
- Check for reflective surfaces

**Cursor jumps:**
- Increase `smoothing_window` value
- Enable Kalman filtering
- Check for camera shake

### Performance Issues

**Low FPS:**
- Use `--low-end` mode
- Close background applications
- Reduce camera resolution

**High CPU usage:**
- Increase `process_every_n_frames`
- Use lite MediaPipe model (default)
- Reduce `max_hands` to 1

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     GestureControlSystem                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Main Loop                              │  │
│  │  1. Capture frame from webcam                            │  │
│  │  2. Preprocess (flip, resize, lighting adjust)           │  │
│  │  3. Detect hand landmarks (MediaPipe)                    │  │
│  │  4. Extract hand state (finger positions, gestures)      │  │
│  │  5. Recognize gesture (pattern matching)                 │  │
│  │  6. Validate gesture (stability, confidence)             │  │
│  │  7. Execute action (PyAutoGUI)                           │  │
│  │  8. Render UI overlay                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Components:                                                    │
│  ├── HandDetector (MediaPipe integration)                      │
│  ├── GestureRecognizer (pattern matching)                      │
│  ├── ActionExecutor (PyAutoGUI wrapper)                        │
│  ├── SmoothingFilter (position stabilization)                  │
│  ├── GestureDebouncer (action rate limiting)                   │
│  └── CalibrationManager (user calibration)                     │
└─────────────────────────────────────────────────────────────────┘
```

## Extending the System

### Adding New Gestures

```python
# 1. Add to GestureType enum
class GestureType(Enum):
    MY_GESTURE = auto()

# 2. Add recognition logic in GestureRecognizer
def _is_my_gesture(self, state: HandState) -> bool:
    return (state.index_extended and 
            state.middle_extended and 
            state.ring_extended and
            not state.pinky_extended)

# 3. Add to recognize() priority list
if self._is_my_gesture(state):
    return GestureType.MY_GESTURE

# 4. Add action handler in ActionExecutor
def _handle_my_gesture(self, state: HandState) -> str:
    pyautogui.hotkey('ctrl', 'c')  # Example: copy
    return "Copied!"

# 5. Register in action_map
self.action_map[GestureType.MY_GESTURE] = self._handle_my_gesture
```

### Custom Actions

```python
# In ActionExecutor, add custom handlers:

def _handle_custom_screenshot(self, state: HandState) -> str:
    """Custom screenshot gesture."""
    if self.debouncer.can_trigger("screenshot", 1000):
        pyautogui.hotkey('win', 'shift', 's')  # Windows snip
        return "Screenshot!"
    return "Screenshot (cooldown)"
```

## File Structure

```
__handguestures__/
├── gesture_control.py     # Main system implementation
├── advanced_processing.py # Noise reduction, filtering
├── utils.py               # Utilities and profiles
├── run.py                 # Quick start script
├── tests.py               # Unit tests
├── requirements.txt       # Dependencies
├── README.md              # System documentation
└── USAGE_GUIDE.md         # This file
```

## License

MIT License - Free for personal and commercial use.
