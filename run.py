"""
Quick Start Script for Hand Gesture Control
Runs system check and starts the gesture controller.
"""

import sys
import os

def main():
    print("=" * 60)
    print("  HAND GESTURE LAPTOP CONTROL SYSTEM")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("[1/3] Checking dependencies...")
    missing = []
    
    try:
        import cv2
        print(f"  [OK] OpenCV {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")
        print("  [X] OpenCV not found")
    
    try:
        import mediapipe as mp
        mp_version = getattr(mp, '__version__', 'unknown')
        print(f"  [OK] MediaPipe {mp_version}")
    except ImportError:
        missing.append("mediapipe")
        print("  [X] MediaPipe not found")
    
    try:
        import pyautogui
        print(f"  [OK] PyAutoGUI {pyautogui.__version__}")
    except ImportError:
        missing.append("pyautogui")
        print("  [X] PyAutoGUI not found")
    
    try:
        import numpy as np
        print(f"  [OK] NumPy {np.__version__}")
    except ImportError:
        missing.append("numpy")
        print("  [X] NumPy not found")
    
    if missing:
        print()
        print("Missing dependencies! Install with:")
        print(f"  pip install {' '.join(missing)}")
        print()
        print("Or run: pip install -r requirements.txt")
        return 1
    
    # Check camera
    print()
    print("[2/3] Checking camera...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [X] Cannot access camera!")
        print("  Make sure your webcam is connected and not in use by another app.")
        return 1
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"  [OK] Camera available: {width}x{height} @ {fps:.0f}fps")
    
    # Print controls
    print()
    print("[3/3] Starting gesture control...")
    print()
    print("+" + "=" * 60 + "+")
    print("|" + " " * 20 + "GESTURE CONTROLS" + " " * 24 + "|")
    print("+" + "=" * 60 + "+")
    print("|  Open Palm    -> Move cursor                              |")
    print("|  Pinch        -> Left click                               |")
    print("|  Thumb+Middle -> Right click                              |")
    print("|  Victory (V)  -> Scroll (move up/down)                    |")
    print("|  Fist         -> Pause/Resume                             |")
    print("|  Thumbs Up    -> Play/Pause media                         |")
    print("+" + "=" * 60 + "+")
    print("|  KEYBOARD: Q=Quit  C=Calibrate  P=Pause                   |")
    print("+" + "=" * 60 + "+")
    print()
    
    # Start the main controller
    from gesture_control import GestureControlSystem, GestureConfig
    
    # Detect if running on low-end system
    config = GestureConfig()
    
    # Auto-detect performance mode
    if fps < 25 or width < 640:
        print("[!] Lower camera specs detected, enabling performance mode...")
        config.process_every_n_frames = 2
        config.frame_width = 480
        config.frame_height = 360
    
    system = GestureControlSystem(config)
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    print("\nGoodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
