"""
Interview Intelligence System - Face Clustering & Role Identification

Handles multi-person face detection, clustering, and role assignment 
using a simple greedy clustering algorithm to avoid external dependencies.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import mediapipe as mp
    # Check which API is available: new Tasks API vs legacy solutions API
    MEDIAPIPE_SOLUTIONS_AVAILABLE = hasattr(mp, 'solutions')
    MEDIAPIPE_TASKS_AVAILABLE = hasattr(mp, 'tasks')
    MEDIAPIPE_AVAILABLE = MEDIAPIPE_SOLUTIONS_AVAILABLE or MEDIAPIPE_TASKS_AVAILABLE
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_SOLUTIONS_AVAILABLE = False
    MEDIAPIPE_TASKS_AVAILABLE = False
    
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class IdentityData:
    id: str  # "person_0", "person_1", etc.
    role: str = "unknown"  # "interviewee", "interviewer_1", etc.
    face_encoding: np.ndarray = field(default_factory=lambda: np.array([]))
    num_appearances: int = 0
    representative_image: Optional[np.ndarray] = None

class FaceClusterer:
    """
    Clusters faces from a video to identify unique individuals.
    """
    
    def __init__(self, tolerance: float = 0.6):
        """
        Initialize face clusterer.
        """
        self.tolerance = tolerance
        self.identities: Dict[int, IdentityData] = {}

    def process_video_for_clustering(self, video_path: str, sample_rate: int = 30) -> Dict[int, IdentityData]:
        """
        Scans video, extracts face embeddings, clusters them to find unique people.
        """
        if not FACE_RECOGNITION_AVAILABLE and not MEDIAPIPE_AVAILABLE:
            logger.warning("Neither face_recognition nor mediapipe found. Falling back to single identity.")
            return {0: IdentityData(id="person_0", num_appearances=100)}
            
        if not FACE_RECOGNITION_AVAILABLE:
            return self._process_with_mediapipe_fallback(video_path, sample_rate)
            
        logger.info(f"Starting face clustering on {video_path}")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        all_encodings = []
        all_frames_with_faces = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    for i, encoding in enumerate(encodings):
                        all_encodings.append(encoding)
                        top, right, bottom, left = face_locations[i]
                        face_image = frame[top:bottom, left:right].copy()
                        all_frames_with_faces.append(face_image)

            frame_count += 1
        cap.release()
        
        if not all_encodings:
            return {}

        # Simple Greedy Clustering
        clusters = [] # List of lists of indices
        
        for i, encoding in enumerate(all_encodings):
            matched = False
            for cluster in clusters:
                # Compare with a representative element or average
                # For simplicity, compare with the first element of the cluster
                dist = np.linalg.norm(encoding - all_encodings[cluster[0]])
                if dist < self.tolerance:
                    cluster.append(i)
                    matched = True
                    break
            if not matched:
                clusters.append([i])
        
        logger.info(f"Identified {len(clusters)} unique clusters using greedy method")
        
        for label_id, indices in enumerate(clusters):
            # Only consider clusters with a few appearances to filter noise
            if len(indices) < 2 and len(clusters) > 1:
                continue
                
            cluster_encodings = [all_encodings[i] for i in indices]
            avg_encoding = np.mean(cluster_encodings, axis=0)
            rep_image = all_frames_with_faces[indices[0]]
            
            self.identities[label_id] = IdentityData(
                id=f"person_{label_id}",
                face_encoding=avg_encoding,
                num_appearances=len(indices),
                representative_image=rep_image
            )
            
        return self.identities

    def auto_assign_roles(self, identities: Dict[int, IdentityData]) -> Dict[str, IdentityData]:
        """
        Assign roles based on appearance frequency.
        """
        if not identities:
            return {}
            
        sorted_ids = sorted(identities.values(), key=lambda x: x.num_appearances, reverse=True)
        if sorted_ids:
            sorted_ids[0].role = "interviewee"
        for i, identity in enumerate(sorted_ids[1:]):
            identity.role = f"interviewer_{i+1}"
            
        return {p.id: p for p in sorted_ids}

    def _process_with_mediapipe_fallback(self, video_path: str, sample_rate: int = 30) -> Dict[int, IdentityData]:
        """
        Uses MediaPipe for face detection and clusters based on spatial/size heuristics.
        Reliable for identifying 'Candidate' vs 'Interviewer' in static shots.
        Supports both legacy mp.solutions API and new mp.tasks API.
        """
        logger.info(f"Using MediaPipe fallback for clustering on {video_path}")
        
        if MEDIAPIPE_SOLUTIONS_AVAILABLE:
            return self._process_with_mediapipe_solutions(video_path, sample_rate)
        elif MEDIAPIPE_TASKS_AVAILABLE:
            return self._process_with_mediapipe_tasks(video_path, sample_rate)
        else:
            logger.warning("No MediaPipe API available. Falling back to single identity.")
            return {0: IdentityData(id="person_0", num_appearances=100)}

    def _process_with_mediapipe_solutions(self, video_path: str, sample_rate: int = 30) -> Dict[int, IdentityData]:
        """Legacy mp.solutions based face detection."""
        mp_face_detection = mp.solutions.face_detection
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        raw_clusters = []
        
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_count % sample_rate == 0:
                    h, w, _ = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)
                    
                    if results.detections:
                        for det in results.detections:
                            bbox = det.location_data.relative_bounding_box
                            self._add_detection_to_clusters(raw_clusters, bbox, frame, h, w)
                
                frame_count += 1
        cap.release()
        
        return self._build_identities_from_clusters(raw_clusters)

    def _process_with_mediapipe_tasks(self, video_path: str, sample_rate: int = 30) -> Dict[int, IdentityData]:
        """New mp.tasks based face detection (mediapipe >= 0.10)."""
        import os
        import urllib.request
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        
        # Download the model if not present
        model_path = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")
        if not os.path.exists(model_path):
            logger.info("Downloading MediaPipe face detection model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
            try:
                urllib.request.urlretrieve(url, model_path)
                logger.info("Model downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download face detection model: {e}")
                return {0: IdentityData(id="person_0", num_appearances=100)}
        
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5
        )
        
        detector = mp_vision.FaceDetector.create_from_options(options)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        raw_clusters = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % sample_rate == 0:
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = detector.detect(mp_image)
                
                if result.detections:
                    for det in result.detections:
                        bbox = det.bounding_box
                        # Convert pixel bbox to relative coordinates matching legacy API
                        class RelativeBBox:
                            def __init__(self, xmin, ymin, width, height):
                                self.xmin = xmin
                                self.ymin = ymin
                                self.width = width
                                self.height = height
                        
                        rel_bbox = RelativeBBox(
                            xmin=bbox.origin_x / w,
                            ymin=bbox.origin_y / h,
                            width=bbox.width / w,
                            height=bbox.height / h
                        )
                        self._add_detection_to_clusters(raw_clusters, rel_bbox, frame, h, w)
            
            frame_count += 1
        cap.release()
        detector.close()
        
        return self._build_identities_from_clusters(raw_clusters)

    def _add_detection_to_clusters(self, raw_clusters: list, bbox, frame: np.ndarray, h: int, w: int):
        """Add a face detection to the appropriate cluster based on centroid proximity."""
        cx = bbox.xmin + bbox.width / 2
        cy = bbox.ymin + bbox.height / 2
        size = bbox.width * bbox.height
        
        detection_data = {
            "centroid": np.array([cx, cy]),
            "size": size,
            "bbox": (
                int(bbox.ymin * h),
                int((bbox.xmin + bbox.width) * w),
                int((bbox.ymin + bbox.height) * h),
                int(bbox.xmin * w)
            ),
            "frame": frame.copy()
        }
        
        matched = False
        for cluster in raw_clusters:
            cluster_centroids = np.array([d["centroid"] for d in cluster])
            avg_centroid = np.mean(cluster_centroids, axis=0)
            dist = np.linalg.norm(detection_data["centroid"] - avg_centroid)
            
            if dist < 0.15:
                cluster.append(detection_data)
                matched = True
                break
        
        if not matched:
            raw_clusters.append([detection_data])

    def _build_identities_from_clusters(self, raw_clusters: list) -> Dict[int, IdentityData]:
        """Convert raw face detection clusters into IdentityData objects."""
        for i, cluster in enumerate(raw_clusters):
            if len(cluster) < 2 and len(raw_clusters) > 1:
                continue
            
            rep_det = max(cluster, key=lambda x: x["size"])
            top, right, bottom, left = rep_det["bbox"]
            top, bottom = max(0, top), min(rep_det["frame"].shape[0], bottom)
            left, right = max(0, left), min(rep_det["frame"].shape[1], right)
            face_crop = rep_det["frame"][top:bottom, left:right].copy()
            
            self.identities[i] = IdentityData(
                id=f"person_{i}",
                num_appearances=len(cluster),
                representative_image=face_crop
            )
        
        logger.info(f"MediaPipe fallback identified {len(self.identities)} individuals.")
        return self.identities

