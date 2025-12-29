"""
Configuration and utility modules for Hand Gesture Control System.
"""

from dataclasses import dataclass, asdict
import json
import os
from typing import Optional


@dataclass
class GestureProfile:
    """User-specific gesture profile with calibrated thresholds."""
    name: str = "default"
    pinch_threshold: float = 0.05
    finger_extend_threshold: float = 0.03
    movement_sensitivity: float = 1.5
    scroll_sensitivity: float = 50
    click_debounce_ms: int = 300
    
    def save(self, path: str):
        """Save profile to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'GestureProfile':
        """Load profile from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class PerformanceMonitor:
    """Monitors system performance and suggests optimizations."""
    
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_times = []
        self.max_samples = 100
    
    def record_frame_time(self, duration_ms: float):
        """Record frame processing time."""
        self.frame_times.append(duration_ms)
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)
    
    def get_average_fps(self) -> float:
        """Calculate average FPS from frame times."""
        if not self.frame_times:
            return 0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1000 / avg_time if avg_time > 0 else 0
    
    def get_recommendations(self) -> list:
        """Get performance optimization recommendations."""
        recommendations = []
        avg_fps = self.get_average_fps()
        
        if avg_fps < self.target_fps * 0.5:
            recommendations.append("Critical: Consider using --low-end mode")
            recommendations.append("Reduce camera resolution")
        elif avg_fps < self.target_fps * 0.75:
            recommendations.append("Consider processing every 2nd frame")
            recommendations.append("Reduce detection confidence threshold")
        
        return recommendations


# Gesture reference guide
GESTURE_GUIDE = """
╔══════════════════════════════════════════════════════════════════════╗
║                    HAND GESTURE CONTROL GUIDE                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  CURSOR CONTROL                                                      ║
║  ─────────────                                                       ║
║  🖐️ Open Palm    : Move cursor (all 5 fingers extended)              ║
║  ☝️ Index Point  : Precise cursor control (only index extended)      ║
║                                                                      ║
║  CLICKING                                                            ║
║  ────────                                                            ║
║  👌 Pinch        : Left click (thumb + index touch)                  ║
║  👌 Pinch Hold   : Click and drag (hold pinch > 0.3s)               ║
║  🤏 Two-Finger   : Right click (thumb + middle touch)               ║
║                                                                      ║
║  SCROLLING                                                           ║
║  ─────────                                                           ║
║  ✌️ Victory      : Scroll mode (move hand up/down to scroll)        ║
║                                                                      ║
║  MEDIA CONTROLS                                                      ║
║  ──────────────                                                      ║
║  👍 Thumbs Up    : Play/Pause media                                  ║
║  👎 Thumbs Down  : Volume down                                       ║
║  🤘 Rock Sign    : Next track                                        ║
║                                                                      ║
║  SYSTEM                                                              ║
║  ──────                                                              ║
║  ✊ Fist         : Pause/Resume control (safety stop)               ║
║                                                                      ║
║  KEYBOARD SHORTCUTS                                                  ║
║  ──────────────────                                                  ║
║  Q : Quit application                                                ║
║  C : Run calibration                                                 ║
║  P : Pause/Resume                                                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def print_guide():
    """Print the gesture guide to console."""
    print(GESTURE_GUIDE)


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import mediapipe
    except ImportError:
        missing.append("mediapipe")
    
    try:
        import pyautogui
    except ImportError:
        missing.append("pyautogui")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    return True


def get_camera_info(camera_index: int = 0) -> dict:
    """Get information about available camera."""
    import cv2
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"error": f"Cannot open camera {camera_index}"}
    
    info = {
        "index": camera_index,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "backend": cap.getBackendName(),
    }
    
    cap.release()
    return info
