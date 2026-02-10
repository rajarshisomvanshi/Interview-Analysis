"""
Interview Intelligence System - Facial Analysis

Implements face recognition, facial landmarks, eye contact, blink detection, and Action Units.
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

try:
    import mediapipe as mp
except ImportError:
    mp = None
    logging.warning("mediapipe not installed. Install with: pip install mediapipe")

from config import settings
from core.schemas import FaceSignal

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Face recognition for identity locking.
    
    Uses face embeddings to identify and track the interviewee throughout the session.
    """
    
    def __init__(self):
        """Initialize face recognizer"""
        self.reference_embedding: Optional[np.ndarray] = None
        self.identity_locked = False
        
        logger.info("Initialized FaceRecognizer")
        
    def extract_embedding(self, frame: np.ndarray, face_box: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from a detected face.
        
        Args:
            frame: Input frame (BGR format)
            face_box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Face embedding vector or None if extraction fails
        """

        # Crop face region
        x1, y1, x2, y2 = map(int, face_box)
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
            
        # Convert to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Simple embedding: use resized face as feature vector (placeholder)
        # In production, use FaceNet, ArcFace, or similar
        face_resized = cv2.resize(face_rgb, (128, 128))
        embedding = face_resized.flatten().astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
        
        return embedding
        
    def lock_identity(self, frame: np.ndarray, face_box: np.ndarray) -> bool:
        """
        Lock identity using the first clear face detection.
        
        Args:
            frame: Input frame
            face_box: Face bounding box
            
        Returns:
            True if identity locked successfully
        """
        embedding = self.extract_embedding(frame, face_box)
        
        if embedding is not None:
            self.reference_embedding = embedding
            self.identity_locked = True
            logger.info("Identity locked successfully")
            return True
            
        return False
        
    def verify_identity(self, frame: np.ndarray, face_box: np.ndarray) -> Optional[float]:
        """
        Verify if detected face matches locked identity.
        
        Args:
            frame: Input frame
            face_box: Face bounding box
            
        Returns:
            Confidence score (0-1) or None if verification fails
        """
        if not self.identity_locked or self.reference_embedding is None:
            return None
            
        embedding = self.extract_embedding(frame, face_box)
        
        if embedding is None:
            return None
            
        # Compute cosine similarity
        similarity = np.dot(self.reference_embedding, embedding)
        confidence = float((similarity + 1) / 2)  # Map from [-1, 1] to [0, 1]
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        return confidence


class FacialLandmarkExtractor:
    """
    Extract 468-point facial landmarks using MediaPipe Face Landmarker (Tasks API).
    """
    
    def __init__(self):
        """Initialize facial landmark extractor"""
        if mp is None:
            logging.warning("MediaPipe not available. Landmark extraction disabled.")
            self.landmarker = None
        else:
            try:
                import mediapipe.tasks.python as tasks
                import mediapipe.tasks.python.vision as vision
                
                base_options = tasks.BaseOptions(model_asset_path='models/face_landmarker.task')
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=1,
                    running_mode=vision.RunningMode.VIDEO
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.mp_image_format = mp.ImageFormat.SRGB
                logger.info("Initialized FacialLandmarkExtractor (Tasks API)")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe Tasks: {e}")
                self.landmarker = None
        
    def extract_landmarks(self, frame: np.ndarray, timestamp_ms: int) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
        """
        Extract facial landmarks and blendshapes from frame.
        
        Args:
            frame: Input frame (BGR format)
            timestamp_ms: Framework timestamp in milliseconds
            
        Returns:
            Tuple of (landmarks_array, blendshapes_dict)
            landmarks_array: [468, 3] (x, y, z)
            blendshapes_dict: Dict of {category_name: score} (52 blendshapes)
        """
        if self.landmarker is None:
            return None, None

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MP Image
        mp_image = mp.Image(image_format=self.mp_image_format, data=frame_rgb)
        
        # Process frame
        try:
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            logger.warning(f"MediaPipe inference failed: {e}")
            return None, None
        
        if not detection_result.face_landmarks:
            return None, None
            
        # Get first face landmarks
        landmarks = detection_result.face_landmarks[0]
        
        # Convert landmarks to numpy array (denormalize)
        h, w = frame.shape[:2]
        landmarks_array = np.array([
            [lm.x * w, lm.y * h, lm.z]
            for lm in landmarks
        ])
        
        # Get blendshapes if available
        blendshapes_dict = {}
        if detection_result.face_blendshapes:
            # face_blendshapes is a list of lists (one per face)
            # each item has category_name and score
            for category in detection_result.face_blendshapes[0]:
                blendshapes_dict[category.category_name] = category.score
                
        return landmarks_array, blendshapes_dict
        
    def get_eye_landmarks(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract eye region landmarks.
        
        Args:
            landmarks: Full landmarks array [468, 3]
            
        Returns:
            Dictionary with 'left_eye' and 'right_eye' landmark indices
        """
        # MediaPipe eye landmark indices remain the same for Face Mesh / Landmarker
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        return {
            'left_eye': landmarks[left_eye_indices],
            'right_eye': landmarks[right_eye_indices]
        }


class EyeContactDetector:
    """
    Detect eye contact and gaze direction.
    """
    
    def __init__(self):
        """Initialize eye contact detector"""
        self.landmark_extractor = FacialLandmarkExtractor()
        logger.info("Initialized EyeContactDetector")
        
    def compute_eye_contact(self, landmarks: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Compute eye contact score and gaze direction.
        
        Args:
            landmarks: Facial landmarks [468, 3]
            
        Returns:
            Tuple of (eye_contact_score, gaze_direction_dict)
        """
        # Simplified gaze estimation using nose and eye landmarks
        nose_tip = landmarks[1]  # Nose tip
        left_eye_center = landmarks[33]  # Left eye center
        right_eye_center = landmarks[263]  # Right eye center
        
        # Compute eye center
        eye_center = (left_eye_center + right_eye_center) / 2
        
        # Compute gaze vector (simplified)
        gaze_vector = nose_tip - eye_center
        gaze_vector = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-8)
        
        # Eye contact score: higher when looking forward (z-component close to 0)
        eye_contact_score = 1.0 - abs(gaze_vector[2])
        eye_contact_score = np.clip(eye_contact_score, 0.0, 1.0)
        
        gaze_direction = {
            'x': float(gaze_vector[0]),
            'y': float(gaze_vector[1]),
            'z': float(gaze_vector[2])
        }
        
        return float(eye_contact_score), gaze_direction


class BlinkDetector:
    """
    Detect blinks using Eye Aspect Ratio (EAR).
    """
    
    EAR_THRESHOLD = 0.25  # Threshold for blink detection (increased from 0.21)
    
    def __init__(self):
        """Initialize blink detector"""
        self.landmark_extractor = FacialLandmarkExtractor()
        logger.info("Initialized BlinkDetector")
        
    def compute_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Compute Eye Aspect Ratio.
        
        Args:
            eye_landmarks: Eye landmarks [6, 3]
            
        Returns:
            EAR value
        """
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR formula
        ear = (v1 + v2) / (2.0 * h + 1e-8)
        
        return ear
        
    def detect_blink(self, landmarks: np.ndarray, blendshapes: Optional[Dict[str, float]] = None) -> Tuple[float, float, bool]:
        """
        Detect blink in current frame.
        
        Args:
            landmarks: Facial landmarks [468, 3]
            blendshapes: Optional MediaPipe blendshapes
            
        Returns:
            Tuple of (left_eye_openness, right_eye_openness, blink_detected)
        """
        # Strategy 1: Use Blendshapes (Superior)
        if blendshapes:
            # MediaPipe outputs 'eyeBlinkLeft' and 'eyeBlinkRight'
            # Value is 0 (open) to 1 (closed)
            blink_left = blendshapes.get('eyeBlinkLeft', 0.0)
            blink_right = blendshapes.get('eyeBlinkRight', 0.0)
            
            # Convert to openness (1 = open, 0 = closed)
            left_openness = 1.0 - blink_left
            right_openness = 1.0 - blink_right
            
            # Detect blink using AVERAGE closedness to handle asymmetry/ptosis
            # Threshold: 0.5 (meaning eyes are on average 50% closed)
            avg_blink = (blink_left + blink_right) / 2.0
            blink_detected = (avg_blink > 0.5)
            
            return float(np.clip(left_openness, 0.0, 1.0)), float(np.clip(right_openness, 0.0, 1.0)), blink_detected
            
        # Strategy 2: Fallback to EAR (Legacy/Backup)
        # Get eye landmarks
        eye_landmarks = self.landmark_extractor.get_eye_landmarks(landmarks)
        
        # Compute EAR for both eyes
        left_ear = self.compute_ear(eye_landmarks['left_eye'])
        right_ear = self.compute_ear(eye_landmarks['right_eye'])
        
        # Normalize to 0-1 range (higher = more open)
        left_openness = min(left_ear / self.EAR_THRESHOLD, 1.0)
        right_openness = min(right_ear / self.EAR_THRESHOLD, 1.0)
        
        # Detect blink if both eyes below threshold
        blink_detected = (left_ear < self.EAR_THRESHOLD and right_ear < self.EAR_THRESHOLD)
        
        return float(left_openness), float(right_openness), blink_detected


class ActionUnitExtractor:
    """
    Extract Action Units (AUs) from facial landmarks.
    
    Simplified AU extraction based on landmark movements.
    """
    
    def __init__(self):
        """Initialize Action Unit extractor"""
        self.landmark_extractor = FacialLandmarkExtractor()
        self.baseline_landmarks: Optional[np.ndarray] = None
        logger.info("Initialized ActionUnitExtractor")
        
    def set_baseline(self, landmarks: np.ndarray):
        """Set baseline landmarks for neutral expression"""
        self.baseline_landmarks = landmarks.copy()
        
    def extract_action_units(self, landmarks: np.ndarray, blendshapes: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Extract Action Unit intensities.
        
        Args:
            landmarks: Current facial landmarks [468, 3]
            blendshapes: Optional MediaPipe blendshapes
            
        Returns:
            Dictionary of AU intensities (0-1 scale)
        """
        # Strategy 1: Use Blendshapes (Superior)
        if blendshapes:
            aus = {}
            # AU1: Inner brow raiser
            aus['AU1'] = blendshapes.get('browInnerUp', 0.0)
            
            # AU4: Brow lowerer (avg of left/right)
            aus['AU4'] = (blendshapes.get('browDownLeft', 0.0) + blendshapes.get('browDownRight', 0.0)) / 2.0
            
            # AU12: Lip corner puller (smile)
            aus['AU12'] = (blendshapes.get('mouthSmileLeft', 0.0) + blendshapes.get('mouthSmileRight', 0.0)) / 2.0
            
            # AU15: Lip corner depressor (frown)
            aus['AU15'] = (blendshapes.get('mouthFrownLeft', 0.0) + blendshapes.get('mouthFrownRight', 0.0)) / 2.0
            
            # AU26: Jaw drop
            aus['AU26'] = blendshapes.get('jawOpen', 0.0)
            
            # Clip all values to 0-1 range
            for k, v in aus.items():
                aus[k] = float(np.clip(v, 0.0, 1.0))
            
            return aus

        # Strategy 2: Fallback to Heuristics
        if self.baseline_landmarks is None:
            self.set_baseline(landmarks)
            
        # Compute displacement from baseline
        displacement = landmarks - self.baseline_landmarks
        
        # Simplified AU extraction based on landmark geometric ratios
        aus = {}
        
        h, w = self.baseline_landmarks.shape[:2] if self.baseline_landmarks is not None else (1, 1)

        # AU1: Inner brow raiser
        # AU4: Brow lowerer
        # Using vertical displacement of inner brow landmarks relative to nose
        brow_inner = [70, 300]
        brow_disp = np.mean(displacement[brow_inner, 1]) # y-axis
        aus['AU1'] = float(np.clip(-brow_disp / 10.0, 0, 1)) # Negative displacement is "up"
        aus['AU4'] = float(np.clip(brow_disp / 10.0, 0, 1))

        # AU12: Lip corner puller (smile)
        mouth_corners = [61, 291]
        mouth_disp_y = np.mean(displacement[mouth_corners, 1])
        aus['AU12'] = float(np.clip(-mouth_disp_y / 15.0, 0, 1))
        
        # AU15: Lip corner depressor (frown)
        aus['AU15'] = float(np.clip(mouth_disp_y / 15.0, 0, 1))

        # AU26: Jaw drop
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        jaw_dist = np.linalg.norm(upper_lip - lower_lip)
        aus['AU26'] = float(np.clip(jaw_dist / 30.0, 0, 1))

        return aus


class EmotionPriorClassifier:
    """
    Lightweight emotion classifier providing 'weak priors' based on AU intensities.
    """
    
    def classify(self, aus: Dict[str, float]) -> Dict[str, float]:
        """
        Produce probabilistic emotion labels based on AU scores.
        """
        emotions = {
            'happy': aus.get('AU12', 0) * 0.8,
            'sad': aus.get('AU15', 0) * 0.7 + aus.get('AU4', 0) * 0.3,
            'surprised': aus.get('AU26', 0) * 0.6 + aus.get('AU1', 0) * 0.4,
            'neutral': 1.0 - max(aus.values()) if aus else 1.0
        }
        
        # Normalize
        total = sum(emotions.values()) + 1e-8
        return {k: v / total for k, v in emotions.items()}


class FacialAnalyzer:
    """
    Complete facial analysis pipeline combining all components.
    """
    
    def __init__(self):
        """Initialize facial analyzer"""
        self.recognizer = FaceRecognizer()
        self.landmark_extractor = FacialLandmarkExtractor()
        self.eye_contact_detector = EyeContactDetector()
        self.blink_detector = BlinkDetector()
        self.au_extractor = ActionUnitExtractor()
        self.emotion_classifier = EmotionPriorClassifier()
        
        logger.info("Initialized FacialAnalyzer")
        
    def analyze_frame(
        self,
        frame: np.ndarray,
        face_box: Optional[np.ndarray],
        timestamp_ms: int,
        interviewer_box: Optional[np.ndarray] = None
    ) -> FaceSignal:
        """
        Perform complete facial analysis on a single frame.
        
        Args:
            frame: Input frame (BGR)
            face_box: Face bounding box [x1, y1, x2, y2] or None
            timestamp_ms: Frame timestamp in milliseconds
            
        Returns:
            FaceSignal object with all extracted signals
        """
        signal = FaceSignal(timestamp_ms=timestamp_ms, face_detected=False)
        
        if face_box is None:
            return signal
            
        signal.face_detected = True
        
        # Face recognition
        if not self.recognizer.identity_locked:
            self.recognizer.lock_identity(frame, face_box)
            signal.identity_confidence = 1.0
        else:
            conf = self.recognizer.verify_identity(frame, face_box)
            signal.identity_confidence = float(np.clip(conf, 0.0, 1.0)) if conf is not None else None
            
        # Extract landmarks and blendshapes
        landmarks, blendshapes = self.landmark_extractor.extract_landmarks(frame, timestamp_ms)
        
        if landmarks is not None:
            # Eye contact
            # print("DEBUG: Computing eye contact")
            eye_contact, gaze_dir = self.eye_contact_detector.compute_eye_contact(landmarks)
            signal.eye_contact = float(np.clip(eye_contact, 0.0, 1.0))
            signal.gaze_direction = gaze_dir
            
            # Blink detection
            # print("DEBUG: Detecting blink")
            left_open, right_open, blink = self.blink_detector.detect_blink(landmarks, blendshapes)
            signal.left_eye_open = left_open
            signal.right_eye_open = right_open
            signal.blink_detected = blink
            
            # Action Units
            # print("DEBUG: Extracting AUs")
            signal.action_units = self.au_extractor.extract_action_units(landmarks, blendshapes)
            
            # Emotion Weak Priors
            # print("DEBUG: Classifying emotions")
            signal.emotions = self.emotion_classifier.classify(signal.action_units)
            
            # Gaze towards Interviewer
            if interviewer_box is not None:
                # print("DEBUG: Calculating gaze adherence")
                # Simple check: is the gaze vector pointing roughly towards the center of the interviewer box?
                # This is a heuristic: if x-gaze and y-gaze align with direction to interviewer
                face_center = np.mean(face_box.reshape(2, 2), axis=0)
                interviewer_center = np.mean(interviewer_box.reshape(2, 2), axis=0)
                dir_to_interviewer = interviewer_center - face_center
                # Normalize dir
                dir_to_interviewer = dir_to_interviewer / (np.linalg.norm(dir_to_interviewer) + 1e-8)
                
                # Check alignment with gaze_dir (x, y)
                alignment = np.dot(dir_to_interviewer, [gaze_dir['x'], gaze_dir['y']])
                signal.action_units['gaze_towards_interviewer'] = float(np.clip(alignment, 0, 1))
            
            # Store landmarks
            signal.landmarks = [{'x': float(lm[0]), 'y': float(lm[1]), 'z': float(lm[2])} for lm in landmarks]
            
        return signal
