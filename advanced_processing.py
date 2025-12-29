"""
Advanced gesture detection with noise reduction and lighting compensation.
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, List
from dataclasses import dataclass
import cv2


@dataclass
class FrameStats:
    """Statistics about the current frame for adaptive processing."""
    brightness: float = 0.0
    contrast: float = 0.0
    noise_level: float = 0.0
    

class AdaptivePreprocessor:
    """
    Handles frame preprocessing with adaptive adjustments for:
    - Lighting variation
    - Camera noise
    - Exposure changes
    """
    
    def __init__(self, history_size: int = 30):
        self.brightness_history = deque(maxlen=history_size)
        self.target_brightness = 127  # Target average brightness
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def analyze_frame(self, frame: np.ndarray) -> FrameStats:
        """Analyze frame statistics for adaptive processing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        stats = FrameStats()
        stats.brightness = np.mean(gray)
        stats.contrast = np.std(gray)
        
        # Estimate noise using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        stats.noise_level = laplacian.var()
        
        self.brightness_history.append(stats.brightness)
        
        return stats
    
    def preprocess(self, frame: np.ndarray, stats: Optional[FrameStats] = None) -> np.ndarray:
        """
        Apply adaptive preprocessing to improve hand detection.
        
        Techniques applied:
        1. Brightness normalization
        2. Contrast enhancement (CLAHE)
        3. Noise reduction (bilateral filter for edge preservation)
        """
        if stats is None:
            stats = self.analyze_frame(frame)
        
        processed = frame.copy()
        
        # 1. Adaptive brightness adjustment
        if stats.brightness < 80:  # Dark frame
            # Increase brightness
            alpha = min(1.5, self.target_brightness / max(stats.brightness, 1))
            processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=20)
        elif stats.brightness > 180:  # Overexposed frame
            # Reduce brightness
            alpha = self.target_brightness / stats.brightness
            processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=-20)
        
        # 2. Apply CLAHE for contrast enhancement in low-contrast conditions
        if stats.contrast < 40:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 3. Noise reduction for noisy frames (preserves edges)
        if stats.noise_level > 500:
            processed = cv2.bilateralFilter(processed, 5, 50, 50)
        
        return processed
    
    def get_average_brightness(self) -> float:
        """Get rolling average brightness."""
        if not self.brightness_history:
            return self.target_brightness
        return sum(self.brightness_history) / len(self.brightness_history)


class KalmanPositionFilter:
    """
    Kalman filter for smooth position tracking with prediction.
    Reduces jitter while maintaining responsiveness.
    """
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        # State: [x, y, vx, vy] (position and velocity)
        self.state = np.zeros(4)
        self.covariance = np.eye(4) * 1000
        
        # State transition matrix (assumes constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Process noise
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise
        self.R = np.eye(2) * measurement_noise
        
        self.initialized = False
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Update filter with new measurement and return filtered position."""
        measurement = np.array([x, y])
        
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return x, y
        
        # Predict
        predicted_state = self.F @ self.state
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # Update
        innovation = measurement - self.H @ predicted_state
        innovation_covariance = self.H @ predicted_covariance @ self.H.T + self.R
        kalman_gain = predicted_covariance @ self.H.T @ np.linalg.inv(innovation_covariance)
        
        self.state = predicted_state + kalman_gain @ innovation
        self.covariance = (np.eye(4) - kalman_gain @ self.H) @ predicted_covariance
        
        return float(self.state[0]), float(self.state[1])
    
    def predict(self) -> Tuple[float, float]:
        """Predict next position without measurement (for occlusion handling)."""
        predicted = self.F @ self.state
        return float(predicted[0]), float(predicted[1])
    
    def reset(self):
        """Reset filter state."""
        self.state = np.zeros(4)
        self.covariance = np.eye(4) * 1000
        self.initialized = False


class GestureStateManager:
    """
    Manages gesture state transitions with hysteresis and validation.
    Prevents rapid switching between gestures (flickering).
    """
    
    def __init__(self, stability_threshold: int = 3, hysteresis_frames: int = 2):
        self.stability_threshold = stability_threshold
        self.hysteresis_frames = hysteresis_frames
        
        self.current_gesture = None
        self.pending_gesture = None
        self.gesture_count = 0
        self.frames_since_change = 0
    
    def update(self, detected_gesture) -> tuple:
        """
        Update gesture state with stability checking.
        Returns (stable_gesture, is_new_gesture)
        """
        is_new = False
        
        if detected_gesture == self.pending_gesture:
            self.gesture_count += 1
        else:
            self.pending_gesture = detected_gesture
            self.gesture_count = 1
        
        # Gesture becomes stable after threshold frames
        if self.gesture_count >= self.stability_threshold:
            if self.pending_gesture != self.current_gesture:
                # Additional hysteresis: don't change too quickly
                if self.frames_since_change >= self.hysteresis_frames:
                    self.current_gesture = self.pending_gesture
                    self.frames_since_change = 0
                    is_new = True
        
        self.frames_since_change += 1
        
        return self.current_gesture, is_new
    
    def reset(self):
        """Reset state manager."""
        self.current_gesture = None
        self.pending_gesture = None
        self.gesture_count = 0
        self.frames_since_change = 0


class OutlierRejector:
    """
    Rejects outlier positions that are likely due to detection errors.
    Uses statistical analysis of recent positions.
    """
    
    def __init__(self, window_size: int = 10, threshold_std: float = 3.0):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.history_x = deque(maxlen=window_size)
        self.history_y = deque(maxlen=window_size)
    
    def validate(self, x: float, y: float) -> Tuple[bool, float, float]:
        """
        Validate position and return (is_valid, corrected_x, corrected_y).
        If invalid, returns the corrected (interpolated) position.
        """
        if len(self.history_x) < 3:
            self.history_x.append(x)
            self.history_y.append(y)
            return True, x, y
        
        # Calculate statistics
        mean_x = np.mean(self.history_x)
        mean_y = np.mean(self.history_y)
        std_x = np.std(self.history_x) + 1e-6
        std_y = np.std(self.history_y) + 1e-6
        
        # Check if position is an outlier
        z_x = abs(x - mean_x) / std_x
        z_y = abs(y - mean_y) / std_y
        
        is_valid = z_x < self.threshold_std and z_y < self.threshold_std
        
        if is_valid:
            self.history_x.append(x)
            self.history_y.append(y)
            return True, x, y
        else:
            # Return interpolated position
            corrected_x = mean_x + (x - mean_x) / max(z_x, 1)
            corrected_y = mean_y + (y - mean_y) / max(z_y, 1)
            return False, corrected_x, corrected_y
    
    def reset(self):
        """Reset history."""
        self.history_x.clear()
        self.history_y.clear()


class LandmarkValidator:
    """
    Validates landmark detection quality and confidence.
    """
    
    @staticmethod
    def validate_hand_landmarks(landmarks, min_confidence: float = 0.7) -> Tuple[bool, List[str]]:
        """
        Validate hand landmarks for quality.
        Returns (is_valid, list_of_issues).
        """
        issues = []
        
        if landmarks is None:
            return False, ["No landmarks detected"]
        
        # Check if all landmarks are present
        if len(landmarks) < 21:
            issues.append(f"Missing landmarks: {21 - len(landmarks)}")
        
        # Check for landmarks outside frame bounds
        out_of_bounds = 0
        for lm in landmarks:
            if lm.x < 0 or lm.x > 1 or lm.y < 0 or lm.y > 1:
                out_of_bounds += 1
        
        if out_of_bounds > 0:
            issues.append(f"Landmarks out of bounds: {out_of_bounds}")
        
        # Check for anatomically impossible configurations
        # (e.g., finger tip behind palm)
        wrist_y = landmarks[0].y
        for i in [4, 8, 12, 16, 20]:  # Finger tips
            if landmarks[i].y > wrist_y + 0.3:
                issues.append("Anatomically invalid hand pose")
                break
        
        return len(issues) == 0, issues
    
    @staticmethod
    def estimate_occlusion(landmarks) -> float:
        """
        Estimate hand occlusion level (0 = fully visible, 1 = fully occluded).
        """
        if landmarks is None:
            return 1.0
        
        # Check visibility of key landmarks
        key_landmarks = [0, 4, 8, 12, 16, 20]  # Wrist and finger tips
        visible_count = 0
        
        for idx in key_landmarks:
            lm = landmarks[idx]
            # Landmark is considered visible if within frame bounds
            if 0.05 < lm.x < 0.95 and 0.05 < lm.y < 0.95:
                visible_count += 1
        
        return 1.0 - (visible_count / len(key_landmarks))
