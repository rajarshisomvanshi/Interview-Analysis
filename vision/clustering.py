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
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("face_recognition library not found. Falling back to single identity.")
            return {0: IdentityData(id="person_0", num_appearances=100)}
            
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
