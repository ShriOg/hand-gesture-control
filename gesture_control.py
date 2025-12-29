"""
Hand Gesture Laptop Control System
===================================
A real-time hand gesture recognition system for controlling laptop functions
using the built-in webcam.

Compatible with MediaPipe 0.10.x (Tasks API)

Author: Hand Gesture Control Project
License: MIT
"""

import cv2
import pyautogui
import numpy as np
import time
import os
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable PyAutoGUI fail-safe for edge movements (we implement our own safety)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01  # Minimal delay between actions

# MediaPipe imports - handle both old and new API
USE_NEW_API = False
try:
    import mediapipe as mp
    # Check if new Tasks API is available
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    USE_NEW_API = True
    logger.info("Using MediaPipe Tasks API (0.10.x)")
except ImportError:
    try:
        import mediapipe as mp
        # Check for legacy solutions API
        if hasattr(mp, 'solutions'):
            USE_NEW_API = False
            logger.info("Using MediaPipe Legacy API")
        else:
            USE_NEW_API = True
            logger.info("Using MediaPipe Tasks API (0.10.x)")
    except ImportError:
        raise ImportError("MediaPipe is not installed. Run: pip install mediapipe")


class GestureType(Enum):
    """Enumeration of recognized gestures."""
    NONE = auto()
    OPEN_PALM = auto()
    INDEX_POINT = auto()
    PINCH = auto()
    PINCH_HOLD = auto()
    TWO_FINGER_PINCH = auto()
    VICTORY_SCROLL = auto()
    FIST = auto()
    THUMBS_UP = auto()
    THUMBS_DOWN = auto()
    ROCK_SIGN = auto()


@dataclass
class GestureConfig:
    """Configuration parameters for gesture detection."""
    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps_target: int = 30
    
    # Processing settings
    process_every_n_frames: int = 1
    max_hands: int = 1
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.7
    
    # Smoothing settings
    smoothing_window: int = 5
    gesture_stability_frames: int = 3
    
    # Debounce settings
    click_debounce_ms: int = 300
    scroll_debounce_ms: int = 50
    media_debounce_ms: int = 500
    
    # Gesture thresholds
    pinch_threshold: float = 0.05
    finger_extend_threshold: float = 0.03
    
    # Screen mapping
    screen_margin: float = 0.1
    movement_sensitivity: float = 1.5
    scroll_sensitivity: float = 50
    
    # Safety settings
    action_cooldown_ms: int = 100
    max_actions_per_second: int = 30
    hold_time_for_drag: float = 0.3


@dataclass 
class LandmarkPoint:
    """Simple landmark point with x, y, z coordinates."""
    x: float
    y: float
    z: float = 0.0


@dataclass
class HandState:
    """Represents the current state of a detected hand."""
    landmarks: Optional[List[LandmarkPoint]] = None
    handedness: str = "Right"
    confidence: float = 0.0
    
    # Finger states (True = extended)
    thumb_extended: bool = False
    index_extended: bool = False
    middle_extended: bool = False
    ring_extended: bool = False
    pinky_extended: bool = False
    
    # Key positions (normalized 0-1)
    wrist_pos: Tuple[float, float] = (0.0, 0.0)
    index_tip_pos: Tuple[float, float] = (0.0, 0.0)
    thumb_tip_pos: Tuple[float, float] = (0.0, 0.0)
    middle_tip_pos: Tuple[float, float] = (0.0, 0.0)
    palm_center: Tuple[float, float] = (0.0, 0.0)
    
    # Derived values
    pinch_distance: float = 1.0
    two_finger_pinch_distance: float = 1.0
    hand_openness: float = 0.0


class SmoothingFilter:
    """Exponential moving average filter for position smoothing."""
    
    def __init__(self, window_size: int = 5, alpha: float = 0.3):
        self.window_size = window_size
        self.alpha = alpha
        self.history_x: deque = deque(maxlen=window_size)
        self.history_y: deque = deque(maxlen=window_size)
        self.ema_x: Optional[float] = None
        self.ema_y: Optional[float] = None
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        self.history_x.append(x)
        self.history_y.append(y)
        
        if self.ema_x is None:
            self.ema_x = x
            self.ema_y = y
        else:
            self.ema_x = self.alpha * x + (1 - self.alpha) * self.ema_x
            self.ema_y = self.alpha * y + (1 - self.alpha) * self.ema_y
        
        return self.ema_x, self.ema_y
    
    def reset(self):
        self.history_x.clear()
        self.history_y.clear()
        self.ema_x = None
        self.ema_y = None


class GestureDebouncer:
    """Handles debouncing of gesture actions."""
    
    def __init__(self):
        self.last_action_time: Dict[str, float] = {}
        self.gesture_start_time: Dict[GestureType, float] = {}
        self.gesture_frame_count: Dict[GestureType, int] = {}
        self.action_count_window: deque = deque(maxlen=100)
    
    def can_trigger(self, action: str, debounce_ms: int) -> bool:
        current_time = time.time() * 1000
        last_time = self.last_action_time.get(action, 0)
        
        if current_time - last_time >= debounce_ms:
            self.last_action_time[action] = current_time
            self.action_count_window.append(current_time)
            return True
        return False
    
    def is_gesture_stable(self, gesture: GestureType, required_frames: int) -> bool:
        return self.gesture_frame_count.get(gesture, 0) >= required_frames
    
    def update_gesture_stability(self, gesture: GestureType):
        for g in GestureType:
            if g == gesture:
                self.gesture_frame_count[g] = self.gesture_frame_count.get(g, 0) + 1
            else:
                self.gesture_frame_count[g] = 0
    
    def get_gesture_hold_time(self, gesture: GestureType) -> float:
        if gesture not in self.gesture_start_time:
            self.gesture_start_time[gesture] = time.time()
        return time.time() - self.gesture_start_time[gesture]
    
    def reset_gesture_hold(self, gesture: GestureType):
        if gesture in self.gesture_start_time:
            del self.gesture_start_time[gesture]
    
    def get_actions_per_second(self) -> float:
        current_time = time.time() * 1000
        return sum(1 for t in self.action_count_window if current_time - t < 1000)


class HandDetector:
    """MediaPipe-based hand detection - supports both old and new API."""
    
    # Landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
    
    # Hand connections for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),  # Palm
    ]
    
    def __init__(self, config: GestureConfig):
        self.config = config
        self.use_new_api = USE_NEW_API
        
        if self.use_new_api:
            self._init_new_api()
        else:
            self._init_legacy_api()
    
    def _init_new_api(self):
        """Initialize using MediaPipe Tasks API (0.10.x)."""
        # Import here to avoid issues if not available
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        
        # Download model if not present
        model_path = self._ensure_model_exists()
        
        # WORKAROUND: MediaPipe Tasks has a bug with non-ASCII characters in cwd
        # Save current directory and change to temp dir for initialization
        original_cwd = os.getcwd()
        safe_cwd = os.path.dirname(model_path)  # Use temp dir which has ASCII path
        
        try:
            os.chdir(safe_cwd)
            
            # Create hand landmarker
            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.config.max_hands,
                min_hand_detection_confidence=self.config.detection_confidence,
                min_hand_presence_confidence=self.config.detection_confidence,
                min_tracking_confidence=self.config.tracking_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            logger.info("MediaPipe Tasks HandLandmarker initialized")
        finally:
            os.chdir(original_cwd)
    
    def _init_legacy_api(self):
        """Initialize using legacy MediaPipe API."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_hands,
            min_detection_confidence=self.config.detection_confidence,
            min_tracking_confidence=self.config.tracking_confidence,
            model_complexity=0
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def _ensure_model_exists(self) -> str:
        """Download hand landmarker model if not present."""
        import tempfile
        import shutil
        
        # Use temp directory to avoid Unicode path issues with MediaPipe
        temp_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
        os.makedirs(temp_dir, exist_ok=True)
        model_path = os.path.join(temp_dir, "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            logger.info("Downloading hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                logger.info(f"Model downloaded to {model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise RuntimeError(
                    f"Could not download hand landmarker model. "
                    f"Please download manually from {url} and place in {temp_dir}"
                )
        else:
            logger.info(f"Using cached model from {model_path}")
        
        return model_path
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[HandState], np.ndarray]:
        """Detect hand and extract state from frame."""
        if self.use_new_api:
            return self._detect_new_api(frame)
        else:
            return self._detect_legacy_api(frame)
    
    def _detect_new_api(self, frame: np.ndarray) -> Tuple[Optional[HandState], np.ndarray]:
        """Detection using Tasks API."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        results = self.detector.detect(mp_image)
        
        if not results.hand_landmarks:
            return None, frame
        
        # Get first hand's landmarks
        hand_landmarks = results.hand_landmarks[0]
        handedness = results.handedness[0][0] if results.handedness else None
        
        # Convert to our LandmarkPoint format
        landmarks = [
            LandmarkPoint(x=lm.x, y=lm.y, z=lm.z) 
            for lm in hand_landmarks
        ]
        
        # Extract hand state
        hand_state = self._extract_hand_state(
            landmarks,
            handedness.category_name if handedness else "Right",
            handedness.score if handedness else 0.9
        )
        
        # Draw landmarks
        annotated_frame = self._draw_landmarks_manual(frame, landmarks)
        
        return hand_state, annotated_frame
    
    def _detect_legacy_api(self, frame: np.ndarray) -> Tuple[Optional[HandState], np.ndarray]:
        """Detection using legacy API."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, frame
        
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0]
        
        # Convert to our LandmarkPoint format
        landmarks = [
            LandmarkPoint(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks.landmark
        ]
        
        hand_state = self._extract_hand_state(
            landmarks,
            handedness.label,
            handedness.score
        )
        
        # Draw using MediaPipe's drawing utils
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )
        
        return hand_state, frame
    
    def _extract_hand_state(self, landmarks: List[LandmarkPoint], 
                           handedness: str, confidence: float) -> HandState:
        """Extract comprehensive hand state from landmarks."""
        state = HandState()
        state.landmarks = landmarks
        state.handedness = handedness
        state.confidence = confidence
        
        # Extract key positions
        state.wrist_pos = (landmarks[self.WRIST].x, landmarks[self.WRIST].y)
        state.index_tip_pos = (landmarks[self.INDEX_TIP].x, landmarks[self.INDEX_TIP].y)
        state.thumb_tip_pos = (landmarks[self.THUMB_TIP].x, landmarks[self.THUMB_TIP].y)
        state.middle_tip_pos = (landmarks[self.MIDDLE_TIP].x, landmarks[self.MIDDLE_TIP].y)
        
        # Calculate palm center
        palm_x = np.mean([landmarks[self.INDEX_MCP].x, landmarks[self.MIDDLE_MCP].x,
                         landmarks[self.RING_MCP].x, landmarks[self.PINKY_MCP].x])
        palm_y = np.mean([landmarks[self.INDEX_MCP].y, landmarks[self.MIDDLE_MCP].y,
                         landmarks[self.RING_MCP].y, landmarks[self.PINKY_MCP].y])
        state.palm_center = (palm_x, palm_y)
        
        # Detect finger extension states
        state.thumb_extended = self._is_thumb_extended(landmarks, handedness)
        state.index_extended = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP)
        state.middle_extended = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        state.ring_extended = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP)
        state.pinky_extended = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP)
        
        # Calculate pinch distances
        state.pinch_distance = self._calculate_distance(
            landmarks[self.THUMB_TIP], landmarks[self.INDEX_TIP]
        )
        state.two_finger_pinch_distance = self._calculate_distance(
            landmarks[self.THUMB_TIP], landmarks[self.MIDDLE_TIP]
        )
        
        # Calculate hand openness
        fingers_extended = sum([
            state.thumb_extended, state.index_extended, state.middle_extended,
            state.ring_extended, state.pinky_extended
        ])
        state.hand_openness = fingers_extended / 5.0
        
        return state
    
    def _is_thumb_extended(self, landmarks: List[LandmarkPoint], handedness: str) -> bool:
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        
        if handedness == "Right":
            return thumb_tip.x < thumb_ip.x - self.config.finger_extend_threshold
        else:
            return thumb_tip.x > thumb_ip.x + self.config.finger_extend_threshold
    
    def _is_finger_extended(self, landmarks: List[LandmarkPoint], 
                           tip_idx: int, pip_idx: int) -> bool:
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        return tip.y < pip.y - self.config.finger_extend_threshold
    
    def _calculate_distance(self, p1: LandmarkPoint, p2: LandmarkPoint) -> float:
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def _draw_landmarks_manual(self, frame: np.ndarray, 
                               landmarks: List[LandmarkPoint]) -> np.ndarray:
        """Draw landmarks manually (for Tasks API)."""
        h, w = frame.shape[:2]
        
        # Draw connections
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for lm in landmarks:
            point = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, point, 5, (255, 0, 0), -1)
        
        return frame
    
    def release(self):
        """Release resources."""
        if self.use_new_api:
            if hasattr(self, 'detector'):
                self.detector.close()
        else:
            if hasattr(self, 'hands'):
                self.hands.close()


class GestureRecognizer:
    """Recognizes specific gestures from hand state."""
    
    def __init__(self, config: GestureConfig):
        self.config = config
    
    def recognize(self, state: HandState) -> GestureType:
        if state is None:
            return GestureType.NONE
        
        # Priority-ordered gesture recognition
        if self._is_fist(state):
            return GestureType.FIST
        
        if self._is_pinch(state):
            return GestureType.PINCH
        
        if self._is_two_finger_pinch(state):
            return GestureType.TWO_FINGER_PINCH
        
        if self._is_thumbs_up(state):
            return GestureType.THUMBS_UP
        
        if self._is_thumbs_down(state):
            return GestureType.THUMBS_DOWN
        
        if self._is_rock_sign(state):
            return GestureType.ROCK_SIGN
        
        if self._is_victory(state):
            return GestureType.VICTORY_SCROLL
        
        if self._is_index_point(state):
            return GestureType.INDEX_POINT
        
        if self._is_open_palm(state):
            return GestureType.OPEN_PALM
        
        return GestureType.NONE
    
    def _is_fist(self, state: HandState) -> bool:
        return state.hand_openness < 0.2
    
    def _is_open_palm(self, state: HandState) -> bool:
        return state.hand_openness >= 0.8
    
    def _is_index_point(self, state: HandState) -> bool:
        return (state.index_extended and
                not state.middle_extended and
                not state.ring_extended and
                not state.pinky_extended)
    
    def _is_pinch(self, state: HandState) -> bool:
        return state.pinch_distance < self.config.pinch_threshold
    
    def _is_two_finger_pinch(self, state: HandState) -> bool:
        return (state.two_finger_pinch_distance < self.config.pinch_threshold and
                state.pinch_distance >= self.config.pinch_threshold)
    
    def _is_victory(self, state: HandState) -> bool:
        return (state.index_extended and
                state.middle_extended and
                not state.ring_extended and
                not state.pinky_extended)
    
    def _is_thumbs_up(self, state: HandState) -> bool:
        if not state.thumb_extended:
            return False
        if any([state.index_extended, state.middle_extended,
                state.ring_extended, state.pinky_extended]):
            return False
        # Check thumb pointing upward
        if state.landmarks:
            thumb_tip_y = state.landmarks[4].y
            thumb_ip_y = state.landmarks[3].y
            wrist_y = state.landmarks[0].y
            return thumb_tip_y < thumb_ip_y and thumb_ip_y < wrist_y
        return False
    
    def _is_thumbs_down(self, state: HandState) -> bool:
        if not state.thumb_extended:
            return False
        if any([state.index_extended, state.middle_extended,
                state.ring_extended, state.pinky_extended]):
            return False
        if state.landmarks:
            thumb_tip_y = state.landmarks[4].y
            thumb_ip_y = state.landmarks[3].y
            return thumb_tip_y > thumb_ip_y
        return False
    
    def _is_rock_sign(self, state: HandState) -> bool:
        return (state.index_extended and
                state.pinky_extended and
                not state.middle_extended and
                not state.ring_extended)


class ActionExecutor:
    """Executes system actions based on recognized gestures."""
    
    def __init__(self, config: GestureConfig):
        self.config = config
        self.screen_width, self.screen_height = pyautogui.size()
        self.position_filter = SmoothingFilter(config.smoothing_window)
        self.scroll_filter = SmoothingFilter(config.smoothing_window, alpha=0.2)
        self.debouncer = GestureDebouncer()
        
        self.is_dragging = False
        self.last_scroll_y = 0.5
        self.is_paused = False
        self.last_gesture = GestureType.NONE
        
        self.action_map: Dict[GestureType, Callable] = {
            GestureType.OPEN_PALM: self._handle_cursor_control,
            GestureType.INDEX_POINT: self._handle_cursor_control,
            GestureType.PINCH: self._handle_pinch,
            GestureType.TWO_FINGER_PINCH: self._handle_right_click,
            GestureType.VICTORY_SCROLL: self._handle_scroll,
            GestureType.FIST: self._handle_fist,
            GestureType.THUMBS_UP: self._handle_play_pause,
            GestureType.THUMBS_DOWN: self._handle_volume_down,
            GestureType.ROCK_SIGN: self._handle_next_track,
        }
    
    def execute(self, gesture: GestureType, hand_state: HandState) -> str:
        if self.is_paused and gesture != GestureType.FIST:
            return "PAUSED - Show fist to resume"
        
        if self.debouncer.get_actions_per_second() > self.config.max_actions_per_second:
            return "Rate limited"
        
        self.debouncer.update_gesture_stability(gesture)
        
        if not self.debouncer.is_gesture_stable(gesture, self.config.gesture_stability_frames):
            return f"Stabilizing: {gesture.name}"
        
        handler = self.action_map.get(gesture)
        if handler:
            result = handler(hand_state)
            self.last_gesture = gesture
            return result
        
        if self.last_gesture == GestureType.PINCH and gesture != GestureType.PINCH:
            self._end_drag()
        
        return gesture.name
    
    def _handle_cursor_control(self, state: HandState) -> str:
        raw_x, raw_y = state.index_tip_pos
        smooth_x, smooth_y = self.position_filter.update(raw_x, raw_y)
        
        margin = self.config.screen_margin
        norm_x = (smooth_x - margin) / (1 - 2 * margin)
        norm_y = (smooth_y - margin) / (1 - 2 * margin)
        
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        # Invert X for mirror effect
        screen_x = int((1 - norm_x) * self.screen_width * self.config.movement_sensitivity)
        screen_y = int(norm_y * self.screen_height * self.config.movement_sensitivity)
        
        # Keep away from corners to avoid PyAutoGUI failsafe
        corner_margin = 5
        screen_x = max(corner_margin, min(self.screen_width - corner_margin, screen_x))
        screen_y = max(corner_margin, min(self.screen_height - corner_margin, screen_y))
        
        try:
            pyautogui.moveTo(screen_x, screen_y, _pause=False)
        except pyautogui.FailSafeException:
            return "Failsafe - move away from corner"
        
        return f"Cursor: ({screen_x}, {screen_y})"
    
    def _handle_pinch(self, state: HandState) -> str:
        hold_time = self.debouncer.get_gesture_hold_time(GestureType.PINCH)
        
        if hold_time >= self.config.hold_time_for_drag:
            if not self.is_dragging:
                pyautogui.mouseDown(_pause=False)
                self.is_dragging = True
                return "Drag started"
            else:
                self._handle_cursor_control(state)
                return "Dragging..."
        else:
            return f"Pinch hold: {hold_time:.2f}s"
    
    def _end_drag(self):
        if self.is_dragging:
            pyautogui.mouseUp(_pause=False)
            self.is_dragging = False
        else:
            if self.debouncer.can_trigger("left_click", self.config.click_debounce_ms):
                pyautogui.click(_pause=False)
        
        self.debouncer.reset_gesture_hold(GestureType.PINCH)
    
    def _handle_right_click(self, state: HandState) -> str:
        if self.debouncer.can_trigger("right_click", self.config.click_debounce_ms):
            pyautogui.rightClick(_pause=False)
            return "Right click"
        return "Right click (debounced)"
    
    def _handle_scroll(self, state: HandState) -> str:
        current_y = state.palm_center[1]
        _, smooth_y = self.scroll_filter.update(0.5, current_y)
        
        delta_y = smooth_y - self.last_scroll_y
        
        if abs(delta_y) > 0.01:
            scroll_amount = int(delta_y * self.config.scroll_sensitivity)
            if self.debouncer.can_trigger("scroll", self.config.scroll_debounce_ms):
                pyautogui.scroll(-scroll_amount, _pause=False)
                self.last_scroll_y = smooth_y
                return f"Scroll: {scroll_amount}"
        
        return "Scroll mode"
    
    def _handle_fist(self, state: HandState) -> str:
        if self.debouncer.can_trigger("fist", 1000):
            self.is_paused = not self.is_paused
            if self.is_dragging:
                pyautogui.mouseUp(_pause=False)
                self.is_dragging = False
            return "PAUSED" if self.is_paused else "RESUMED"
        return "Fist (toggle pause)"
    
    def _handle_play_pause(self, state: HandState) -> str:
        if self.debouncer.can_trigger("play_pause", self.config.media_debounce_ms):
            pyautogui.press('playpause', _pause=False)
            return "Play/Pause"
        return "Play/Pause (debounced)"
    
    def _handle_volume_down(self, state: HandState) -> str:
        if self.debouncer.can_trigger("volume_down", self.config.media_debounce_ms):
            pyautogui.press('volumedown', _pause=False)
            return "Volume Down"
        return "Volume Down (debounced)"
    
    def _handle_next_track(self, state: HandState) -> str:
        if self.debouncer.can_trigger("next_track", self.config.media_debounce_ms):
            pyautogui.press('nexttrack', _pause=False)
            return "Next Track"
        return "Next Track (debounced)"
    
    def reset(self):
        self.position_filter.reset()
        self.scroll_filter.reset()
        if self.is_dragging:
            pyautogui.mouseUp(_pause=False)
            self.is_dragging = False


class CalibrationManager:
    """Handles system calibration."""
    
    def __init__(self, config: GestureConfig):
        self.config = config
        self.calibration_data = {
            'pinch_threshold': config.pinch_threshold,
            'finger_extend_threshold': config.finger_extend_threshold,
        }
        self.samples = []
    
    def start_calibration(self):
        self.samples = []
        logger.info("Calibration started.")
    
    def collect_sample(self, state: HandState, gesture_name: str):
        if state:
            self.samples.append({
                'gesture': gesture_name,
                'pinch_distance': state.pinch_distance,
                'hand_openness': state.hand_openness,
            })
    
    def compute_thresholds(self) -> Dict:
        if not self.samples:
            return self.calibration_data
        
        pinch_samples = [s['pinch_distance'] for s in self.samples if s['gesture'] == 'pinch']
        open_samples = [s['pinch_distance'] for s in self.samples if s['gesture'] == 'open']
        
        if pinch_samples and open_samples:
            min_open = min(open_samples)
            max_pinch = max(pinch_samples)
            self.calibration_data['pinch_threshold'] = (min_open + max_pinch) / 2
        
        logger.info(f"Calibration complete: {self.calibration_data}")
        return self.calibration_data


class GestureControlSystem:
    """Main system controller."""
    
    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()
        self.detector = HandDetector(self.config)
        self.recognizer = GestureRecognizer(self.config)
        self.executor = ActionExecutor(self.config)
        self.calibrator = CalibrationManager(self.config)
        
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
    
    def start(self):
        logger.info("Starting Hand Gesture Control System...")
        
        self.cap = cv2.VideoCapture(self.config.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps_target)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        logger.info(f"Camera initialized: {self.config.frame_width}x{self.config.frame_height}")
        self.running = True
        
        try:
            self._main_loop()
        finally:
            self.stop()
    
    def _main_loop(self):
        logger.info("Press 'q' to quit, 'c' to calibrate, 'p' to pause")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                continue
            
            self.frame_count += 1
            
            if self.frame_count % self.config.process_every_n_frames != 0:
                continue
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand
            hand_state, annotated_frame = self.detector.detect(frame)
            
            # Recognize gesture
            gesture = self.recognizer.recognize(hand_state)
            
            # Execute action
            action_result = ""
            if hand_state:
                action_result = self.executor.execute(gesture, hand_state)
            
            # Update FPS
            self._update_fps()
            
            # Draw overlay
            self._draw_overlay(annotated_frame, gesture, action_result, hand_state)
            
            # Display
            cv2.imshow("Hand Gesture Control", annotated_frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('c'):
                self._run_calibration()
            elif key == ord('p'):
                self.executor.is_paused = not self.executor.is_paused
    
    def _update_fps(self):
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = current_time
    
    def _draw_overlay(self, frame: np.ndarray, gesture: GestureType,
                      action_result: str, hand_state: Optional[HandState]):
        height, width = frame.shape[:2]
        
        # Background
        cv2.rectangle(frame, (0, 0), (width, 90), (0, 0, 0), -1)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Gesture
        gesture_color = (0, 255, 255) if gesture != GestureType.NONE else (128, 128, 128)
        cv2.putText(frame, f"Gesture: {gesture.name}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
        
        # Action
        cv2.putText(frame, f"Action: {action_result}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Pause indicator
        if self.executor.is_paused:
            cv2.putText(frame, "PAUSED", (width // 2 - 50, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Finger states
        if hand_state:
            fingers = ["T", "I", "M", "R", "P"]
            states = [hand_state.thumb_extended, hand_state.index_extended,
                     hand_state.middle_extended, hand_state.ring_extended,
                     hand_state.pinky_extended]
            
            for i, (f, s) in enumerate(zip(fingers, states)):
                color = (0, 255, 0) if s else (0, 0, 255)
                cv2.putText(frame, f, (width - 120 + i * 20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Dead zone
            margin = int(self.config.screen_margin * width)
            cv2.rectangle(frame, (margin, margin),
                         (width - margin, height - margin),
                         (100, 100, 100), 1)
        
        # Instructions
        cv2.putText(frame, "Q:Quit  C:Calibrate  P:Pause", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _run_calibration(self):
        logger.info("Starting calibration...")
        self.calibrator.start_calibration()
        
        steps = [
            ("Open your hand fully", "open", 30),
            ("Make a pinch gesture", "pinch", 30),
            ("Make a fist", "fist", 30),
        ]
        
        for instruction, gesture_name, frames in steps:
            logger.info(instruction)
            collected = 0
            
            while collected < frames and self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                hand_state, annotated_frame = self.detector.detect(frame)
                
                if hand_state:
                    self.calibrator.collect_sample(hand_state, gesture_name)
                    collected += 1
                
                cv2.rectangle(annotated_frame, (0, 0), (640, 60), (0, 0, 0), -1)
                cv2.putText(annotated_frame, f"CALIBRATION: {instruction}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Progress: {collected}/{frames}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow("Hand Gesture Control", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        thresholds = self.calibrator.compute_thresholds()
        self.config.pinch_threshold = thresholds['pinch_threshold']
        logger.info("Calibration complete!")
    
    def stop(self):
        logger.info("Shutting down...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        self.detector.release()
        self.executor.reset()
        cv2.destroyAllWindows()
        
        logger.info("Shutdown complete")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hand Gesture Laptop Control")
    parser.add_argument("--low-end", action="store_true",
                        help="Enable low-end PC optimizations")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index to use")
    parser.add_argument("--sensitivity", type=float, default=1.5,
                        help="Mouse movement sensitivity")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = GestureConfig()
    config.camera_index = args.camera
    config.movement_sensitivity = args.sensitivity
    
    if args.low_end:
        logger.info("Low-end mode enabled")
        config.process_every_n_frames = 2
        config.frame_width = 480
        config.frame_height = 360
        config.detection_confidence = 0.6
        config.tracking_confidence = 0.6
    
    system = GestureControlSystem(config)
    
    try:
        system.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
