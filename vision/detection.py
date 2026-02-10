"""
Interview Intelligence System - YOLO-based Detection

Implements object detection for phone video (face/head) and CCTV video (pose estimation).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics not installed. Install with: pip install ultralytics")

from config import settings

logger = logging.getLogger(__name__)


class DetectionResult:
    """Container for detection results"""
    
    def __init__(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        keypoints: Optional[np.ndarray] = None
    ):
        """
        Initialize detection result.
        
        Args:
            boxes: Bounding boxes in format [x1, y1, x2, y2]
            confidences: Confidence scores for each detection
            class_ids: Class IDs for each detection
            keypoints: Optional keypoints for pose estimation [N, K, 3] (x, y, conf)
        """
        self.boxes = boxes
        self.confidences = confidences
        self.class_ids = class_ids
        self.keypoints = keypoints
        
    def __len__(self) -> int:
        return len(self.boxes)
    
    def get_best_detection(self) -> Optional[Dict]:
        """Get detection with highest confidence"""
        if len(self.boxes) == 0:
            return None
            
        best_idx = np.argmax(self.confidences)
        result = {
            'box': self.boxes[best_idx],
            'confidence': float(self.confidences[best_idx]),
            'class_id': int(self.class_ids[best_idx])
        }
        
        if self.keypoints is not None:
            result['keypoints'] = self.keypoints[best_idx]
            
        return result


class PersonAttributeExtractor:
    """
    Extracts visual attributes from person detections (clothing color, bags, etc.)
    for the "Search Brain".
    """
    
    # Simple color ranges for basic attribute extraction if no specialized model is available
    COLORS = {
        'red': ([0, 0, 100], [50, 50, 255]),
        'green': ([0, 100, 0], [50, 255, 50]),
        'blue': ([100, 0, 0], [255, 50, 50]),
        'yellow': ([0, 100, 100], [50, 255, 255]),
        'black': ([0, 0, 0], [50, 50, 50]),
        'white': ([200, 200, 200], [255, 255, 255])
    }

    def __init__(self):
        logger.info("Initialized PersonAttributeExtractor")

    def extract_attributes(self, frame: np.ndarray, box: np.ndarray) -> Dict[str, Any]:
        """
        Extract attributes from a person's bounding box.
        
        Args:
            frame: Full image frame
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary of attributes (e.g., {'top_color': 'black', 'has_bag': True})
        """
        x1, y1, x2, y2 = map(int, box)
        # Ensure coordinates are within bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {}

        # Split crop into upper (top) and lower (bottom) halves for clothing
        ch, cw = crop.shape[:2]
        top_half = crop[0:ch//2, :]
        
        # Determine dominant color (simplified)
        top_color = self._get_dominant_color(top_half)
        
        # Check for bag (simplified: presence of significant dark region near center/sides)
        has_bag = self._detect_bag(crop)
        
        return {
            'top_color': top_color,
            'has_bag': has_bag,
            'confidence': 0.8  # Placeholder
        }

    def _get_dominant_color(self, img: np.ndarray) -> str:
        """Simplified dominant color detection"""
        if img.size == 0: return "unknown"
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        
        # Simple Euclidean distance to predefined colors
        min_dist = float('inf')
        dominant = "unknown"
        
        color_rgb = {
            'black': [30, 30, 30],
            'white': [225, 225, 225],
            'red': [0, 0, 200],
            'blue': [200, 0, 0],
            'green': [0, 180, 0],
            'yellow': [0, 200, 200],
            'gray': [128, 128, 128]
        }
        
        for name, rgb in color_rgb.items():
            dist = np.linalg.norm(avg_color - np.array(rgb))
            if dist < min_dist:
                min_dist = dist
                dominant = name
                
        return dominant

    def _detect_bag(self, img: np.ndarray) -> bool:
        """Heuristic-based bag detection placeholder"""
        # In a real scenario, this would be a specialized classifier
        return False


class PhoneVideoDetector:
    """
    Face and head detection for phone camera video.
    
    Uses YOLOv8n for efficient face detection.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize phone video detector.
        
        Args:
            model_path: Path to YOLO model (default: from settings)
        """
        if YOLO is None:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
            
        model_path = model_path or settings.yolo_phone_model
        self.model = YOLO(model_path)
        self.confidence_threshold = settings.face_confidence_threshold
        
        logger.info(f"Initialized PhoneVideoDetector with model: {model_path}")
        
    def detect_faces(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect faces in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            DetectionResult containing face detections
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)[0]
        
        # Extract detections
        boxes = []
        confidences = []
        class_ids = []
        
        if results.boxes is not None:
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    boxes.append(box.xyxy[0].cpu().numpy())
                    confidences.append(conf)
                    class_ids.append(int(box.cls[0]))
                    
        return DetectionResult(
            boxes=np.array(boxes) if boxes else np.empty((0, 4)),
            confidences=np.array(confidences),
            class_ids=np.array(class_ids)
        )
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """
        Detect faces in a batch of frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of DetectionResult objects
        """
        # Run batch inference
        results = self.model(frames, verbose=False)
        
        detections = []
        for result in results:
            boxes = []
            confidences = []
            class_ids = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= self.confidence_threshold:
                        boxes.append(box.xyxy[0].cpu().numpy())
                        confidences.append(conf)
                        class_ids.append(int(box.cls[0]))
                        
            detections.append(DetectionResult(
                boxes=np.array(boxes) if boxes else np.empty((0, 4)),
                confidences=np.array(confidences),
                class_ids=np.array(class_ids)
            ))
            
        return detections


class CCTVDetector:
    """
    Body pose detection for CCTV camera video.
    
    Uses YOLOv8s-pose for pose estimation with keypoints.
    """
    
    # COCO keypoint indices
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize CCTV detector.
        
        Args:
            model_path: Path to YOLO pose model (default: from settings)
        """
        if YOLO is None:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
            
        model_path = model_path or settings.yolo_cctv_model
        self.model = YOLO(model_path)
        self.confidence_threshold = settings.body_confidence_threshold
        self.attribute_extractor = PersonAttributeExtractor()
        
        logger.info(f"Initialized CCTVDetector with model: {model_path}")
        
    def detect_pose(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect body pose in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            DetectionResult containing pose detections with keypoints
        """
        # Run YOLO pose inference
        results = self.model(frame, verbose=False)[0]
        
        # Extract detections
        boxes = []
        confidences = []
        class_ids = []
        keypoints_list = []
        
        if results.boxes is not None and results.keypoints is not None:
            for i, box in enumerate(results.boxes):
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    boxes.append(box.xyxy[0].cpu().numpy())
                    confidences.append(conf)
                    class_ids.append(int(box.cls[0]))
                    keypoints_list.append(results.keypoints.data[i].cpu().numpy())
                    
        return DetectionResult(
            boxes=np.array(boxes) if boxes else np.empty((0, 4)),
            confidences=np.array(confidences),
            class_ids=np.array(class_ids),
            keypoints=np.array(keypoints_list) if keypoints_list else None
        )
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """
        Detect body pose in a batch of frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of DetectionResult objects
        """
        # Run batch inference
        results = self.model(frames, verbose=False)
        
        detections = []
        for result in results:
            boxes = []
            confidences = []
            class_ids = []
            keypoints_list = []
            
            if result.boxes is not None and result.keypoints is not None:
                for i, box in enumerate(result.boxes):
                    conf = float(box.conf[0])
                    if conf >= self.confidence_threshold:
                        boxes.append(box.xyxy[0].cpu().numpy())
                        confidences.append(conf)
                        class_ids.append(int(box.cls[0]))
                        keypoints_list.append(result.keypoints.data[i].cpu().numpy())
                        
            detections.append(DetectionResult(
                boxes=np.array(boxes) if boxes else np.empty((0, 4)),
                confidences=np.array(confidences),
                class_ids=np.array(class_ids),
                keypoints=np.array(keypoints_list) if keypoints_list else None
            ))
            
        return detections
    
    def get_keypoint_by_name(self, keypoints: np.ndarray, name: str) -> Optional[Tuple[float, float, float]]:
        """
        Get specific keypoint by name.
        
        Args:
            keypoints: Keypoints array [17, 3] (x, y, conf)
            name: Keypoint name (e.g., 'left_wrist')
            
        Returns:
            Tuple of (x, y, confidence) or None if not found
        """
        if name not in self.KEYPOINT_NAMES:
            return None
            
        idx = self.KEYPOINT_NAMES.index(name)
        if idx >= len(keypoints):
            return None
            
        return tuple(keypoints[idx])


class MetadataEmitter:
    """
    Packages detections into compact payloads for Layer 2transmission.
    """
    
    def __init__(self, camera_id: str, regional_hub_url: Optional[str] = None):
        self.camera_id = camera_id
        self.regional_hub_url = regional_hub_url
        logger.info(f"Initialized MetadataEmitter for camera: {camera_id}")

    def emit(self, timestamp_ms: int, detections: List[Dict[str, Any]]):
        """
        Create and (optionally) send metadata payload.
        """
        payload = {
            "camera_id": self.camera_id,
            "timestamp_ms": timestamp_ms,
            "iso_time": datetime.fromtimestamp(timestamp_ms/1000).isoformat(),
            "count": len(detections),
            "detections": detections
        }
        
        # For now, we'll just log it or write to a local buffer
        # In Layer 2, we will implement the actual HTTP/MQTT transmission
        metadata_json = json.dumps(payload)
        logger.debug(f"Emitting Metadata: {metadata_json}")
        
        return payload
