"""
CCTV Intelligence System - Edge Service (Layer 1)

Runs on edge hardware to process camera streams and emit metadata payloads.
"""

import cv2
import time
import logging
from pathlib import Path
from typing import Optional

from vision.detection import CCTVDetector, MetadataEmitter
from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EdgeService")

class EdgeService:
    def __init__(self, camera_id: str, stream_source: str, regional_hub_url: Optional[str] = None):
        self.camera_id = camera_id
        self.stream_source = stream_source
        self.detector = CCTVDetector()
        self.emitter = MetadataEmitter(camera_id, regional_hub_url)
        self.is_running = False

    def start(self, frame_skip: int = 5):
        """Start processing the stream"""
        logger.info(f"Starting Edge Service for camera {self.camera_id} on source {self.stream_source}")
        
        cap = cv2.VideoCapture(self.stream_source)
        if not cap.isOpened():
            logger.error(f"Failed to open stream source: {self.stream_source}")
            return

        self.is_running = True
        frame_count = 0
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("End of stream or failed to read frame.")
                    break
                
                if frame_count % frame_skip == 0:
                    timestamp_ms = int(time.time() * 1000)
                    
                    # 1. Detect Persons
                    detection_result = self.detector.detect_pose(frame)
                    
                    # 2. Extract Attributes for each person
                    detections_metadata = []
                    for i, box in enumerate(detection_result.boxes):
                        attributes = self.detector.attribute_extractor.extract_attributes(frame, box)
                        
                        detections_metadata.append({
                            "person_id": i, # Note: Edge-only ID, not global
                            "box": box.tolist(),
                            "confidence": float(detection_result.confidences[i]),
                            "attributes": attributes
                        })
                    
                    # 3. Emit Metadata
                    if detections_metadata:
                        self.emitter.emit(timestamp_ms, detections_metadata)
                        logger.info(f"Emitted metadata for {len(detections_metadata)} persons")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Edge Service stopping...")
        finally:
            cap.release()
            self.is_running = False
            logger.info("Edge Service stopped.")

if __name__ == "__main__":
    import sys
    
    # Default to test video if no source provided
    source = sys.argv[1] if len(sys.argv) > 1 else "videoplayback (2).mp4"
    camera_id = "temple_gate_01"
    
    service = EdgeService(camera_id, source)
    # Process every 10th frame to simulate edge constraints
    service.start(frame_skip=10)
