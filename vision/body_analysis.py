"""
Interview Intelligence System - Body Movement Analysis

Analyzes body movements from CCTV video using pose keypoints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

from core.schemas import BodySignal
from config import settings

logger = logging.getLogger(__name__)


class MovementTracker:
    """
    Track movement of specific body parts over time.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize movement tracker.
        
        Args:
            window_size: Number of frames to track for movement analysis
        """
        self.window_size = window_size
        self.position_history: deque = deque(maxlen=window_size)
        
    def add_position(self, position: np.ndarray):
        """Add new position to history"""
        self.position_history.append(position)
        
    def compute_movement_intensity(self) -> float:
        """
        Compute movement intensity based on position variance.
        
        Returns:
            Movement intensity score (0 = no movement, higher = more movement)
        """
        if len(self.position_history) < 2:
            return 0.0
            
        positions = np.array(list(self.position_history))
        
        # Compute variance in position
        variance = np.var(positions, axis=0)
        intensity = np.sqrt(np.sum(variance))
        
        return float(intensity)
        
    def detect_shift(self, threshold: float = 50.0) -> bool:
        """
        Detect significant position shift.
        
        Args:
            threshold: Pixel threshold for shift detection
            
        Returns:
            True if significant shift detected
        """
        if len(self.position_history) < 2:
            return False
            
        # Compare current position to mean of history
        current = self.position_history[-1]
        historical_mean = np.mean(list(self.position_history)[:-1], axis=0)
        
        distance = np.linalg.norm(current - historical_mean)
        
        return distance > threshold


class HandMovementAnalyzer:
    """
    Analyze hand movements and fidgeting.
    """
    
    def __init__(self):
        """Initialize hand movement analyzer"""
        self.left_hand_tracker = MovementTracker(window_size=30)
        self.right_hand_tracker = MovementTracker(window_size=30)
        logger.info("Initialized HandMovementAnalyzer")
        
    def analyze(self, keypoints: np.ndarray) -> Dict[str, any]:
        """
        Analyze hand movements from pose keypoints.
        
        Args:
            keypoints: Pose keypoints [17, 3] (x, y, conf)
            
        Returns:
            Dictionary with hand movement metrics
        """
        # Wrist keypoint indices (COCO format)
        LEFT_WRIST = 9
        RIGHT_WRIST = 10
        
        result = {
            'left_hand_position': None,
            'right_hand_position': None,
            'hand_movement_intensity': 0.0
        }
        
        # Extract wrist positions
        if keypoints[LEFT_WRIST, 2] > 0.5:  # Confidence check
            left_pos = keypoints[LEFT_WRIST, :2]
            result['left_hand_position'] = {'x': float(left_pos[0]), 'y': float(left_pos[1])}
            self.left_hand_tracker.add_position(left_pos)
            
        if keypoints[RIGHT_WRIST, 2] > 0.5:
            right_pos = keypoints[RIGHT_WRIST, :2]
            result['right_hand_position'] = {'x': float(right_pos[0]), 'y': float(right_pos[1])}
            self.right_hand_tracker.add_position(right_pos)
            
        # Compute combined movement intensity
        left_intensity = self.left_hand_tracker.compute_movement_intensity()
        right_intensity = self.right_hand_tracker.compute_movement_intensity()
        result['hand_movement_intensity'] = (left_intensity + right_intensity) / 2.0
        
        return result


class PostureAnalyzer:
    """
    Analyze posture and detect posture shifts.
    """
    
    def __init__(self):
        """Initialize posture analyzer"""
        self.torso_tracker = MovementTracker(window_size=60)  # Longer window for posture
        self.shoulder_tracker = MovementTracker(window_size=60)
        logger.info("Initialized PostureAnalyzer")
        
    def compute_torso_angle(self, keypoints: np.ndarray) -> Optional[float]:
        """
        Compute torso angle relative to vertical.
        
        Args:
            keypoints: Pose keypoints [17, 3]
            
        Returns:
            Torso angle in degrees or None
        """
        # Shoulder and hip keypoints
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        
        # Check confidence
        if (keypoints[LEFT_SHOULDER, 2] < 0.5 or keypoints[RIGHT_SHOULDER, 2] < 0.5 or
            keypoints[LEFT_HIP, 2] < 0.5 or keypoints[RIGHT_HIP, 2] < 0.5):
            return None
            
        # Compute shoulder and hip midpoints
        shoulder_mid = (keypoints[LEFT_SHOULDER, :2] + keypoints[RIGHT_SHOULDER, :2]) / 2
        hip_mid = (keypoints[LEFT_HIP, :2] + keypoints[RIGHT_HIP, :2]) / 2
        
        # Compute angle
        torso_vector = shoulder_mid - hip_mid
        angle = np.degrees(np.arctan2(torso_vector[0], torso_vector[1]))
        
        return float(angle)
        
    def analyze(self, keypoints: np.ndarray) -> Dict[str, any]:
        """
        Analyze posture from pose keypoints.
        
        Args:
            keypoints: Pose keypoints [17, 3]
            
        Returns:
            Dictionary with posture metrics
        """
        result = {
            'torso_angle': None,
            'shoulder_position': None,
            'posture_shift_detected': False
        }
        
        # Compute torso angle
        torso_angle = self.compute_torso_angle(keypoints)
        if torso_angle is not None:
            result['torso_angle'] = torso_angle
            self.torso_tracker.add_position(np.array([torso_angle]))
            
        # Get shoulder position
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        
        if keypoints[LEFT_SHOULDER, 2] > 0.5 and keypoints[RIGHT_SHOULDER, 2] > 0.5:
            shoulder_mid = (keypoints[LEFT_SHOULDER, :2] + keypoints[RIGHT_SHOULDER, :2]) / 2
            result['shoulder_position'] = {'x': float(shoulder_mid[0]), 'y': float(shoulder_mid[1])}
            self.shoulder_tracker.add_position(shoulder_mid)
            
            # Detect posture shift
            result['posture_shift_detected'] = self.shoulder_tracker.detect_shift(threshold=30.0)
            
        return result


class LegMovementAnalyzer:
    """
    Analyze leg and foot movements.
    """
    
    def __init__(self):
        """Initialize leg movement analyzer"""
        self.left_leg_tracker = MovementTracker(window_size=30)
        self.right_leg_tracker = MovementTracker(window_size=30)
        logger.info("Initialized LegMovementAnalyzer")
        
    def analyze(self, keypoints: np.ndarray) -> Dict[str, any]:
        """
        Analyze leg movements from pose keypoints.
        
        Args:
            keypoints: Pose keypoints [17, 3]
            
        Returns:
            Dictionary with leg movement metrics
        """
        # Ankle keypoint indices
        LEFT_ANKLE = 15
        RIGHT_ANKLE = 16
        
        result = {
            'leg_movement_intensity': 0.0
        }
        
        # Track ankle positions
        if keypoints[LEFT_ANKLE, 2] > 0.5:
            left_pos = keypoints[LEFT_ANKLE, :2]
            self.left_leg_tracker.add_position(left_pos)
            
        if keypoints[RIGHT_ANKLE, 2] > 0.5:
            right_pos = keypoints[RIGHT_ANKLE, :2]
            self.right_leg_tracker.add_position(right_pos)
            
        # Compute movement intensity
        left_intensity = self.left_leg_tracker.compute_movement_intensity()
        right_intensity = self.right_leg_tracker.compute_movement_intensity()
        result['leg_movement_intensity'] = (left_intensity + right_intensity) / 2.0
        
        return result


class BodyMovementAnalyzer:
    """
    Complete body movement analysis pipeline.
    """
    
    def __init__(self):
        """Initialize body movement analyzer"""
        self.hand_analyzer = HandMovementAnalyzer()
        self.posture_analyzer = PostureAnalyzer()
        self.leg_analyzer = LegMovementAnalyzer()
        logger.info("Initialized BodyMovementAnalyzer")
        
    def analyze_frame(
        self,
        keypoints: Optional[np.ndarray],
        timestamp_ms: int
    ) -> BodySignal:
        """
        Perform complete body movement analysis on a single frame.
        
        Args:
            keypoints: Pose keypoints [17, 3] (x, y, conf) or None
            timestamp_ms: Frame timestamp in milliseconds
            
        Returns:
            BodySignal object with all extracted signals
        """
        signal = BodySignal(timestamp_ms=timestamp_ms, body_detected=False)
        
        if keypoints is None or len(keypoints) == 0:
            return signal
            
        signal.body_detected = True
        
        # Hand movement analysis
        hand_metrics = self.hand_analyzer.analyze(keypoints)
        signal.left_hand_position = hand_metrics['left_hand_position']
        signal.right_hand_position = hand_metrics['right_hand_position']
        signal.hand_movement_intensity = hand_metrics['hand_movement_intensity']
        
        # Posture analysis
        posture_metrics = self.posture_analyzer.analyze(keypoints)
        signal.torso_angle = posture_metrics['torso_angle']
        signal.shoulder_position = posture_metrics['shoulder_position']
        signal.posture_shift_detected = posture_metrics['posture_shift_detected']
        
        # Leg movement analysis
        leg_metrics = self.leg_analyzer.analyze(keypoints)
        signal.leg_movement_intensity = leg_metrics['leg_movement_intensity']
        
        # Store full keypoints
        signal.keypoints = [
            {'x': float(kp[0]), 'y': float(kp[1]), 'confidence': float(kp[2])}
            for kp in keypoints
        ]
        
        return signal
