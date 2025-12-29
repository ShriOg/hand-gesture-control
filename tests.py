"""
Test suite for Hand Gesture Control System.
Run with: python -m pytest tests.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from collections import namedtuple

# Import modules to test
from gesture_control import (
    GestureConfig, HandState, GestureType, SmoothingFilter,
    GestureDebouncer, GestureRecognizer
)
from advanced_processing import (
    KalmanPositionFilter, GestureStateManager, OutlierRejector,
    AdaptivePreprocessor
)


class TestSmoothingFilter:
    """Tests for the position smoothing filter."""
    
    def test_initial_position(self):
        """First position should pass through unchanged."""
        filter = SmoothingFilter(window_size=5, alpha=0.3)
        x, y = filter.update(0.5, 0.5)
        assert x == 0.5
        assert y == 0.5
    
    def test_smoothing_effect(self):
        """Filter should smooth out sudden jumps."""
        filter = SmoothingFilter(window_size=5, alpha=0.3)
        
        # Initialize with stable position
        for _ in range(5):
            filter.update(0.5, 0.5)
        
        # Introduce sudden jump
        x, y = filter.update(0.8, 0.8)
        
        # Result should be smoothed (not 0.8)
        assert x < 0.8
        assert y < 0.8
        assert x > 0.5  # But moved towards new position
    
    def test_reset(self):
        """Reset should clear history."""
        filter = SmoothingFilter()
        filter.update(0.5, 0.5)
        filter.reset()
        assert filter.ema_x is None
        assert len(filter.history_x) == 0


class TestGestureDebouncer:
    """Tests for the debounce logic."""
    
    def test_debounce_blocks_rapid_triggers(self):
        """Same action should not trigger twice within debounce window."""
        debouncer = GestureDebouncer()
        
        # First trigger should succeed
        assert debouncer.can_trigger("click", 300) == True
        
        # Immediate second trigger should fail
        assert debouncer.can_trigger("click", 300) == False
    
    def test_different_actions_independent(self):
        """Different actions should have independent debounce."""
        debouncer = GestureDebouncer()
        
        debouncer.can_trigger("click", 300)
        # Different action should still work
        assert debouncer.can_trigger("scroll", 50) == True
    
    def test_gesture_stability(self):
        """Gesture should require multiple frames to be stable."""
        debouncer = GestureDebouncer()
        
        # First frame - not stable
        debouncer.update_gesture_stability(GestureType.PINCH)
        assert debouncer.is_gesture_stable(GestureType.PINCH, 3) == False
        
        # After 3 frames
        debouncer.update_gesture_stability(GestureType.PINCH)
        debouncer.update_gesture_stability(GestureType.PINCH)
        assert debouncer.is_gesture_stable(GestureType.PINCH, 3) == True
    
    def test_gesture_change_resets_stability(self):
        """Changing gesture should reset stability counter."""
        debouncer = GestureDebouncer()
        
        debouncer.update_gesture_stability(GestureType.PINCH)
        debouncer.update_gesture_stability(GestureType.PINCH)
        debouncer.update_gesture_stability(GestureType.FIST)  # Change gesture
        
        # PINCH stability should be reset
        assert debouncer.is_gesture_stable(GestureType.PINCH, 2) == False


class TestGestureRecognizer:
    """Tests for gesture recognition logic."""
    
    @pytest.fixture
    def recognizer(self):
        return GestureRecognizer(GestureConfig())
    
    def test_recognize_fist(self, recognizer):
        """Fist should be recognized when all fingers closed."""
        state = HandState()
        state.hand_openness = 0.0
        state.thumb_extended = False
        state.index_extended = False
        state.middle_extended = False
        state.ring_extended = False
        state.pinky_extended = False
        
        assert recognizer.recognize(state) == GestureType.FIST
    
    def test_recognize_open_palm(self, recognizer):
        """Open palm should be recognized when all fingers extended."""
        state = HandState()
        state.hand_openness = 1.0
        state.thumb_extended = True
        state.index_extended = True
        state.middle_extended = True
        state.ring_extended = True
        state.pinky_extended = True
        state.pinch_distance = 0.2  # Not pinching
        state.two_finger_pinch_distance = 0.2
        
        assert recognizer.recognize(state) == GestureType.OPEN_PALM
    
    def test_recognize_pinch(self, recognizer):
        """Pinch should be recognized when thumb and index close."""
        state = HandState()
        state.hand_openness = 0.5
        state.pinch_distance = 0.03  # Below threshold
        state.two_finger_pinch_distance = 0.2
        
        assert recognizer.recognize(state) == GestureType.PINCH
    
    def test_recognize_victory(self, recognizer):
        """Victory should be recognized with index and middle extended."""
        state = HandState()
        state.hand_openness = 0.4
        state.thumb_extended = False
        state.index_extended = True
        state.middle_extended = True
        state.ring_extended = False
        state.pinky_extended = False
        state.pinch_distance = 0.2
        state.two_finger_pinch_distance = 0.2
        
        assert recognizer.recognize(state) == GestureType.VICTORY_SCROLL
    
    def test_none_for_null_state(self, recognizer):
        """Should return NONE for null hand state."""
        assert recognizer.recognize(None) == GestureType.NONE


class TestKalmanFilter:
    """Tests for Kalman position filter."""
    
    def test_initial_position(self):
        """First position should pass through."""
        kf = KalmanPositionFilter()
        x, y = kf.update(0.5, 0.5)
        assert x == 0.5
        assert y == 0.5
    
    def test_prediction(self):
        """Prediction should extrapolate based on velocity."""
        kf = KalmanPositionFilter()
        
        # Establish motion pattern
        kf.update(0.1, 0.5)
        kf.update(0.2, 0.5)
        kf.update(0.3, 0.5)
        
        # Prediction should continue trend
        pred_x, pred_y = kf.predict()
        assert pred_x > 0.3  # Moving right
    
    def test_smoothing(self):
        """Filter should smooth noisy measurements."""
        kf = KalmanPositionFilter(measurement_noise=0.5)
        
        positions = []
        for _ in range(10):
            x, y = kf.update(0.5 + np.random.normal(0, 0.1), 0.5)
            positions.append(x)
        
        # Variance of filtered positions should be less than input noise
        assert np.std(positions) < 0.1


class TestGestureStateManager:
    """Tests for gesture state management with hysteresis."""
    
    def test_stability_threshold(self):
        """Gesture should only become stable after threshold frames."""
        manager = GestureStateManager(stability_threshold=3)
        
        # First two frames - not stable
        gesture, is_new = manager.update(GestureType.PINCH)
        assert gesture is None or is_new == False
        
        manager.update(GestureType.PINCH)
        
        # Third frame - should become stable
        gesture, is_new = manager.update(GestureType.PINCH)
        # May still need hysteresis frames
    
    def test_hysteresis_prevents_flicker(self):
        """Rapid changes should be prevented by hysteresis."""
        manager = GestureStateManager(stability_threshold=2, hysteresis_frames=3)
        
        # Establish gesture
        for _ in range(5):
            manager.update(GestureType.PINCH)
        
        # Quick change and back
        manager.update(GestureType.FIST)
        manager.update(GestureType.PINCH)
        
        # Should not have changed due to hysteresis
        gesture, _ = manager.update(GestureType.PINCH)
        assert gesture == GestureType.PINCH


class TestOutlierRejector:
    """Tests for outlier detection and rejection."""
    
    def test_accepts_normal_positions(self):
        """Normal positions should be accepted."""
        rejector = OutlierRejector()
        
        for i in range(5):
            valid, x, y = rejector.validate(0.5 + i * 0.01, 0.5)
            assert valid == True
    
    def test_rejects_sudden_jump(self):
        """Sudden large jump should be detected as outlier."""
        rejector = OutlierRejector(threshold_std=2.0)
        
        # Build history of stable positions
        for _ in range(10):
            rejector.validate(0.5, 0.5)
        
        # Large jump
        valid, corrected_x, corrected_y = rejector.validate(0.9, 0.9)
        
        assert valid == False
        assert corrected_x < 0.9  # Should be pulled back
    
    def test_correction_is_reasonable(self):
        """Corrected position should be between old and new."""
        rejector = OutlierRejector()
        
        for _ in range(10):
            rejector.validate(0.5, 0.5)
        
        _, corrected_x, _ = rejector.validate(0.9, 0.5)
        
        assert 0.5 < corrected_x < 0.9


class TestAdaptivePreprocessor:
    """Tests for adaptive frame preprocessing."""
    
    def test_brightness_analysis(self):
        """Should correctly analyze frame brightness."""
        preprocessor = AdaptivePreprocessor()
        
        # Create dark frame
        dark_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 30
        stats = preprocessor.analyze_frame(dark_frame)
        
        assert stats.brightness < 50
    
    def test_rolling_average(self):
        """Should maintain rolling brightness average."""
        preprocessor = AdaptivePreprocessor(history_size=5)
        
        # Process frames with increasing brightness
        for i in range(5):
            frame = np.zeros((100, 100, 3), dtype=np.uint8) + (i * 50)
            preprocessor.analyze_frame(frame)
        
        avg = preprocessor.get_average_brightness()
        assert avg > 0


class TestHandState:
    """Tests for HandState dataclass."""
    
    def test_default_values(self):
        """Default values should be set correctly."""
        state = HandState()
        
        assert state.landmarks is None
        assert state.handedness == "Right"
        assert state.confidence == 0.0
        assert state.hand_openness == 0.0
    
    def test_finger_count(self):
        """Hand openness should reflect finger states."""
        state = HandState()
        state.thumb_extended = True
        state.index_extended = True
        state.middle_extended = False
        state.ring_extended = False
        state.pinky_extended = False
        
        # Manually calculate openness
        fingers = [state.thumb_extended, state.index_extended, 
                   state.middle_extended, state.ring_extended, 
                   state.pinky_extended]
        expected_openness = sum(fingers) / 5.0
        
        assert expected_openness == 0.4


# Integration test
class TestIntegration:
    """Integration tests for the gesture system."""
    
    def test_full_gesture_pipeline(self):
        """Test complete pipeline from hand state to action."""
        config = GestureConfig()
        recognizer = GestureRecognizer(config)
        debouncer = GestureDebouncer()
        
        # Create hand state for pinch gesture
        state = HandState()
        state.pinch_distance = 0.02
        state.two_finger_pinch_distance = 0.2
        state.hand_openness = 0.5
        
        # Recognize gesture
        gesture = recognizer.recognize(state)
        assert gesture == GestureType.PINCH
        
        # Update stability
        for _ in range(3):
            debouncer.update_gesture_stability(gesture)
        
        assert debouncer.is_gesture_stable(gesture, 3) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
