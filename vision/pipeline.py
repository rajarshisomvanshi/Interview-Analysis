"""
Interview Intelligence System - Vision Processing Pipeline

Orchestrates phone and CCTV video processing with parallel execution.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from vision.detection import PhoneVideoDetector, CCTVDetector
from vision.face_analysis import FacialAnalyzer
from vision.body_analysis import BodyMovementAnalyzer
from vision.clustering import FaceClusterer
from audio.transcription import Transcriber
from reasoning.local_llm import LocalLLMAnalyzer
from core.schemas import FaceSignal, BodySignal, SpeakerLabel, TimeSliceAnalysis
from core.timeline import UnifiedTimeline
from config import settings

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Efficient video reader with frame extraction.
    """
    
    def __init__(self, video_path: Path, target_fps: Optional[int] = None):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
            target_fps: Target FPS for extraction (None = use original FPS)
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_ms = int((self.total_frames / self.original_fps) * 1000)
        
        self.target_fps = target_fps or self.original_fps
        self.frame_skip = max(1, int(self.original_fps / self.target_fps))
        
        logger.info(f"Opened video: {video_path} ({self.original_fps} FPS, {self.total_frames} frames)")
        
    def read_frames(self) -> Tuple[np.ndarray, int]:
        """
        Generator that yields (frame, timestamp_ms) tuples.
        
        Yields:
            Tuple of (frame, timestamp_ms)
        """
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            if frame_idx % self.frame_skip == 0:
                timestamp_ms = int((frame_idx / self.original_fps) * 1000)
                yield frame, timestamp_ms
                
            frame_idx += 1
            
    def release(self):
        """Release video capture"""
        self.cap.release()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VisionPipeline:
    """
    Complete vision processing pipeline for phone and CCTV videos.
    """
    
    def __init__(self):
        """Initialize vision pipeline"""
        logger.info("Initializing PhoneVideoDetector...")
        self.phone_detector = PhoneVideoDetector()
        
        logger.info("Initializing CCTVDetector...")
        self.cctv_detector = CCTVDetector()
        
        logger.info("Initializing FacialAnalyzer...")
        self.facial_analyzer = FacialAnalyzer()
        
        logger.info("Initializing BodyMovementAnalyzer...")
        self.body_analyzer = BodyMovementAnalyzer()
        
        logger.info("Initializing FaceClusterer...")
        self.face_clusterer = FaceClusterer()
        
        self.transcriber = None
        self.local_llm = None
        logger.info("Initializing Transcriber...")
        try:
            self.transcriber = Transcriber()
            logger.info("Transcriber initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Transcriber: {e}")
        
        if settings.use_local_llm:
            try:
                # with open("pipeline_debug.log", "a") as f: f.write(f"DEBUG: VisionPipeline - Initializing LocalLLMAnalyzer...\n")
                logger.info("Initializing LocalLLMAnalyzer...")
                self.local_llm = LocalLLMAnalyzer()
                # with open("pipeline_debug.log", "a") as f: f.write(f"DEBUG: VisionPipeline - LocalLLMAnalyzer initialized successfully.\n")
                logger.info("LocalLLMAnalyzer initialized successfully.")
            except Exception as e:
                with open("pipeline_debug.log", "a") as f: f.write(f"DEBUG: VisionPipeline - Failed to initialize Local LLM: {e}\n")
                logger.error(f"Failed to initialize Local LLM: {e}")
        else:
            with open("pipeline_debug.log", "a") as f: f.write(f"DEBUG: VisionPipeline - use_local_llm is False\n")
        
        logger.info("Initialized VisionPipeline")
        
    def process_phone_video(
        self,
        video_path: Path,
        timeline: UnifiedTimeline,
        save_callback: Optional[Callable[[TimeSliceAnalysis], None]] = None
    ) -> List[FaceSignal]:
        """
        Process phone camera video and extract face signals.
        
        Args:
            video_path: Path to phone video file
            timeline: UnifiedTimeline to add events to
            
        Returns:
            List of FaceSignal objects
        """
        logger.info(f"Processing phone video: {video_path}")
        # with open("pipeline_debug.log", "a") as f: f.write(f"DEBUG: VisionPipeline - Starting process_phone_video for {video_path}\n")
        
        # Step 1: Cluster faces to identify interviewee vs interviewer
        
        # Step 1: Cluster faces to identify interviewee vs interviewer
        # This is a pre-pass to lock identities
        identities = self.face_clusterer.process_video_for_clustering(str(video_path))
        # with open("pipeline_debug.log", "a") as f: f.write(f"DEBUG: VisionPipeline - Clustering complete. Identities found: {len(identities)}\n")
        
        roles = self.face_clusterer.auto_assign_roles(identities) # Cluster ID -> IdentityData
        
        # Identify which cluster IDs correspond to which roles
        interviewee_cluster_id = None
        interviewer_cluster_id = None
        
        for cluster_id, identity in identities.items():
            if identity.role == "interviewee":
                interviewee_cluster_id = cluster_id
            elif identity.role.startswith("interviewer"):
                interviewer_cluster_id = cluster_id
        
        face_signals = []
        
        with VideoReader(video_path, target_fps=settings.video_fps) as reader:
            for frame, timestamp_ms in reader.read_frames():
                # if timestamp_ms > 0 and timestamp_ms % 5000 == 0:
                #      with open("pipeline_debug.log", "a") as f: f.write(f"DEBUG: VisionPipeline - Processing frame at {timestamp_ms}ms\n")
                
                # Process every 3rd frame (effective 10 FPS processing)
                if getattr(self, '_frame_counter', 0) % 3 != 0:
                     self._frame_counter = getattr(self, '_frame_counter', 0) + 1
                     continue
                self._frame_counter = getattr(self, '_frame_counter', 0) + 1
                
                # DEBUG LOG
                if self._frame_counter % 30 == 0:
                    logger.info(f"DEBUG: Processing frame {self._frame_counter} at {timestamp_ms}ms")

                # Detect all faces
                detection = self.phone_detector.detect_faces(frame)
                
                # Match detections to clusters
                # For simplicity, we'll use the clusterer's logic or a simple distance check
                interviewee_box = None
                interviewer_box = None
                
                # If we have clustered identities, we should use them to lock
                # For now, let's use the first two detections if available
                if len(detection.boxes) >= 2:
                    # Sort by box size (interviewee is usually closer/larger)
                    sorted_indices = np.argsort([
                        (b[2]-b[0]) * (b[3]-b[1]) 
                        for b in detection.boxes
                    ])[::-1]
                    
                    interviewee_box = detection.boxes[sorted_indices[0]]
                    interviewer_box = detection.boxes[sorted_indices[1]]
                elif len(detection.boxes) == 1:
                    interviewee_box = detection.boxes[0]
                
                # Analyze interviewee face
                # We pass the interviewer box to monitor gaze direction
                signal = self.facial_analyzer.analyze_frame(
                    frame, interviewee_box, timestamp_ms, interviewer_box
                )
                
                # Add to timeline and results
                timeline.add_event(signal)
                face_signals.append(signal)

                # Interim Analysis Printing (Every 2 minutes)
                current_time_min = timestamp_ms / 60000
                if timestamp_ms - getattr(self, '_last_print_ts', 0) >= 120000:
                    # Determine slice log path
                    slice_log_path = video_path.parent / "2-minute-slices.txt"
                    self._print_interim_analysis(
                        face_signals, timestamp_ms, video_path, 
                        latest_frame=frame, slice_log_path=slice_log_path,
                        save_callback=save_callback
                    )
                    self._last_print_ts = timestamp_ms
                
                if len(face_signals) % 100 == 0:
                    logger.info(f"Processed {len(face_signals)} phone video frames ({current_time_min:.1f} min)")
                    
        logger.info(f"Completed phone video processing: {len(face_signals)} frames")
        return face_signals

    def _print_interim_analysis(self, signals: List[FaceSignal], current_ts: int, video_path: Path, **kwargs):
        """Helper to print interim analysis stats."""
        # Calculate start time for this batch
        last_ts = getattr(self, '_last_print_ts', 0)
        
        # Filter signals for this window
        window_signals = [s for s in signals if s.timestamp_ms > last_ts and s.timestamp_ms <= current_ts]
        
        if not window_signals:
            return None
            
        # Calculate robust averages (filtering None)
        valid_eye_contact = [s.eye_contact for s in window_signals if s.eye_contact is not None]
        avg_eye_contact = np.mean(valid_eye_contact) if valid_eye_contact else 0.0
        
        # Blink rate: blinks per minute
        # Count individual TRUE values
        # We need to count transitions from Open (False) to Closed (True) for blink rate
        blink_count = 0
        was_closed = False
        valid_blink_samples = 0
        for s in window_signals:
            if s.blink_detected is not None:
                valid_blink_samples += 1
                is_closed = s.blink_detected
                if is_closed and not was_closed:
                    blink_count += 1
                was_closed = is_closed
        
        duration_min = (current_ts - last_ts) / 60000.0
        blink_rate = blink_count / duration_min if duration_min > 0 else 0.0
        
        # Dominant emotion
        emotions = []
        for s in window_signals:
            if s.emotions:
                emotions.append(max(s.emotions, key=s.emotions.get))
        
        dominant_emotion = "N/A"
        if emotions:
            from collections import Counter
            dominant_emotion = Counter(emotions).most_common(1)[0][0]
        was_closed = False
        
        for s in window_signals:
            is_closed = s.blink_detected
            if is_closed and not was_closed:
                blink_count += 1
            was_closed = is_closed
            
        duration_min = (current_ts - last_ts) / 60000
        blink_rate = blink_count / duration_min if duration_min > 0 else 0
        
        # Context Analysis (Scene & Dialogue)
        scene_context = {}
        if not hasattr(self, 'scene_analyzer'):
            try:
                from vision.scene_analysis import SceneAnalyzer
                self.scene_analyzer = SceneAnalyzer()
            except Exception as e:
                logger.error(f"Failed to init SceneAnalyzer: {e}")
                self.scene_analyzer = None
                
        # Analyze scene on the LAST frame of the window (representative)
        if self.scene_analyzer:
            # We need the actual frame. In this architecture, we don't cache frames easily.
            # However, we can grab a frame if we passed it or if we just use the last processed one.
            # OPTIMIZATION: We can't easily get the frame here without caching. 
            # Let's extract a frame from video at current_ts using VideoUtils or similar, 
            # OR better, since we are inside the pipeline loop, we should probably run scene analysis 
            # periodically inside process_phone_video loop and store the result in a variable.
            
            # For this interim printing, let's try to extract one frame efficiently.
            from utils.video_utils import VideoUtils
            # Re-using the audio extraction method's logic but for video frame?
            # Actually, let's just use a placeholder or implement frame extraction if critical.
            # Better approach: The `process_phone_video` loop has access to frames. 
            # We should pass `latest_frame` to this method? 
            # Yes, let's update the signature of `_print_interim_analysis` to accept `latest_frame`.
            pass

        # Transcript generation (Full)
        transcript_text = ""
        detected_language = "en" # Default
        if self.transcriber:
            import tempfile
            import os
            from utils.video_utils import VideoUtils
            
            import concurrent.futures
            
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_audio_path = Path(tmp.name)
                
                # Extract audio segment
                if VideoUtils.extract_audio_segment(video_path, temp_audio_path, last_ts, current_ts):
                    # Transcribe with timeout
                    # Returns {'words': [...], 'language': 'en'}
                    logger.info(f"Starting interim transcription for {last_ts}-{current_ts}ms...")
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self.transcriber.transcribe, temp_audio_path)
                        try:
                            # 45 second timeout for transcription of a 2-minute segment
                            transcription_result = future.result(timeout=45.0)
                            transcript_words = transcription_result['words']
                            detected_language = transcription_result.get('language', 'en')
                            transcript_text = " ".join([w['word'] for w in transcript_words])
                            logger.info(f"Interim transcription successful ({len(transcript_text)} chars).")
                        except concurrent.futures.TimeoutError:
                            logger.error("Interim transcription TIMEOUT after 45s.")
                        except Exception as te:
                            logger.error(f"Interim transcription inner error: {te}")
                    
                # Cleanup
                if temp_audio_path.exists():
                    os.unlink(temp_audio_path)
            except Exception as e:
                logger.error(f"Interim transcription wrapper failed: {e}")

        # Analyze Scene if frame available
        if kwargs.get('latest_frame') is not None and self.scene_analyzer:
            scene_context = self.scene_analyzer.analyze_scene(kwargs.get('latest_frame'))

        # Generate description if Local LLM is available
        description = ""
        if self.local_llm:
            signals_dict = {
                'avg_eye_contact': avg_eye_contact,
                'emotions': {dominant_emotion: 1.0},
                'blink_rate': blink_rate,
                'transcript': transcript_text,
                'visual_context': scene_context,  # Pass visual context
                'language': detected_language     # Pass detected language
            }
            logger.info(f"Generating description with Local LLM (Language: {detected_language})...")
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.local_llm.generate_description, signals_dict)
                try:
                    # 60 second timeout for LLM generation
                    description = future.result(timeout=60.0)
                    logger.info("Local LLM description generated successfully.")
                except concurrent.futures.TimeoutError:
                    logger.error("Local LLM description generation TIMEOUT after 60s.")
                    description = "Analysis timeout. Processing continues..."
                except Exception as le:
                    logger.error(f"Local LLM generation failed: {le}")
                    description = f"Analysis error: {le}"
        
        print(f"\n{'='*50}")
        print(f" INTERIM ANALYSIS: {last_ts/60000:.1f}m - {current_ts/60000:.1f}m")
        print(f"{'='*50}")
        print(f" Average Eye Contact: {avg_eye_contact:.2f}")
        print(f" Dominant Emotion:    {dominant_emotion}")
        print(f" Blink Rate:          {blink_rate:.1f} blinks/min")
        if scene_context:
            print(f" Visual Context:      {scene_context}")
        if transcript_text:
             print(f" Spoken content ({len(transcript_text)} chars)")
        if description:
            print(f"{'-'*50}")
            print(f" LLM Analysis:\n{description}")
        print(f"{'='*50}\n")
        
        # Save to 2-minute-slices.txt
        slice_log_path = kwargs.get('slice_log_path')
        if slice_log_path:
            try:
                from datetime import timedelta
                start_time = str(timedelta(milliseconds=int(last_ts)))
                end_time = str(timedelta(milliseconds=int(current_ts)))
                
                with open(slice_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"[{start_time} - {end_time}]\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Average Eye Contact: {avg_eye_contact:.2f}\n")
                    f.write(f"Dominant Emotion: {dominant_emotion}\n")
                    f.write(f"Blink Rate: {blink_rate:.1f} blinks/min\n")
                    if scene_context:
                        f.write(f"Visual Context: {scene_context}\n")
                    if transcript_text:
                        f.write(f"\nTranscript:\n{transcript_text}\n")
                    if description:
                        f.write(f"\nLLM Analysis:\n{description}\n")
                    f.write(f"{'='*80}\n")
                    
                logger.info(f"Saved interim slice to {slice_log_path}")
            except Exception as e:
                logger.error(f"Failed to save interim slice: {e}")

        # Parse and return TimeSliceAnalysis
        scene_desc = None
        dialogue = None
        behavioral = None
        
        # Default scores (neutral fallback)
        llm_fluency = 75.0
        llm_confidence = 75.0
        llm_attitude = 75.0
        score_source = "vision_fallback"

        if description:
            try:
                # Parse Scores (Robust)
                import re
                
                # Search entire description for scores, handling various formats:
                # - Fluency: 85
                # - **Fluency**: 85
                # - **Fluency:** 85
                # - Fluency - 85
                
                # Check for Fluency
                f_match = re.search(r"Fluency\D{0,10}(\d{1,3})", description, re.IGNORECASE)
                if f_match: 
                    llm_fluency = float(f_match.group(1))
                    score_source = "llm"

                # Check for Confidence
                c_match = re.search(r"Confidence\D{0,10}(\d{1,3})", description, re.IGNORECASE)
                if c_match: 
                    llm_confidence = float(c_match.group(1))
                    score_source = "llm"
                    
                # Check for Attitude
                a_match = re.search(r"Attitude\D{0,10}(\d{1,3})", description, re.IGNORECASE)
                if a_match: 
                    llm_attitude = float(a_match.group(1))
                    score_source = "llm"

                # Parse Sections
                if "**Scene Description:**" in description:
                   parts = description.split("**Scene Description:**")[1]
                   scene_desc = parts.split("**Reconstructed Dialogue")[0].strip()
                
                if "**Reconstructed Dialogue" in description:
                   parts = description.split("**Reconstructed Dialogue")[1]
                   if "(Q&A Format):**" in parts:
                        parts = parts.split("(Q&A Format):**")[1]
                   elif "**" in parts: # Handle variations
                        parts = parts.split("**")[1]
                   dialogue = parts.split("**Behavioral Analysis:**")[0].strip()
                   
                if "**Behavioral Analysis:**" in description:
                   parts = description.split("**Behavioral Analysis:**")[1]
                   behavioral = parts.strip()
                    
            except Exception as e:
                logger.warning(f"Failed to parse LLM description: {e}")

        # Final Score Calculation (Weighted Average of LLM Scores)
        # 40% Confidence, 30% Fluency, 30% Attitude
        final_score = round((llm_confidence * 0.4) + (llm_fluency * 0.3) + (llm_attitude * 0.3), 1)
            
        slice_analysis = TimeSliceAnalysis(
            start_ms=int(last_ts),
            end_ms=int(current_ts),
            insight=description[:100] + "..." if description else "Analysis taking place...",
            summary=description or "Processing behavioral signals...",
            scene_description=scene_desc,
            dialogue=dialogue,
            behavioral_analysis=behavioral,
            score=final_score,
            fluency=llm_fluency, 
            confidence=llm_confidence,
            attitude=llm_attitude
        )
        
        # Invoke callback if provided
        save_callback = kwargs.get('save_callback')
        if save_callback:
            try:
                save_callback(slice_analysis)
                logger.info("Invoked save_callback with interim analysis.")
            except Exception as e:
                logger.error(f"Failed to invoke save_callback: {e}")

        return slice_analysis

        
    def process_cctv_video(
        self,
        video_path: Path,
        timeline: UnifiedTimeline
    ) -> List[BodySignal]:
        """
        Process CCTV video and extract body signals.
        
        Args:
            video_path: Path to CCTV video file
            timeline: UnifiedTimeline to add events to
            
        Returns:
            List of BodySignal objects
        """
        logger.info(f"Processing CCTV video: {video_path}")
        body_signals = []
        
        with VideoReader(video_path, target_fps=settings.video_fps) as reader:
            for frame, timestamp_ms in reader.read_frames():
                # Detect pose
                detection = self.cctv_detector.detect_pose(frame)
                best_pose = detection.get_best_detection()
                
                # Analyze body movement
                keypoints = best_pose['keypoints'] if best_pose else None
                signal = self.body_analyzer.analyze_frame(keypoints, timestamp_ms)
                
                # Add to timeline and results
                timeline.add_event(signal)
                body_signals.append(signal)
                
                if len(body_signals) % 100 == 0:
                    logger.info(f"Processed {len(body_signals)} CCTV frames")
                    
        logger.info(f"Completed CCTV video processing: {len(body_signals)} frames")
        return body_signals
        
    def process_videos_parallel(
        self,
        phone_video_path: Path,
        cctv_video_path: Path,
        timeline: UnifiedTimeline
    ) -> Tuple[List[FaceSignal], List[BodySignal]]:
        """
        Process both videos in parallel.
        
        Args:
            phone_video_path: Path to phone video
            cctv_video_path: Path to CCTV video
            timeline: UnifiedTimeline to add events to
            
        Returns:
            Tuple of (face_signals, body_signals)
        """
        logger.info("Starting parallel video processing")
        
        face_signals = []
        body_signals = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both processing tasks
            phone_future = executor.submit(
                self.process_phone_video,
                phone_video_path,
                timeline
            )
            cctv_future = executor.submit(
                self.process_cctv_video,
                cctv_video_path,
                timeline
            )
            
            # Wait for completion
            for future in as_completed([phone_future, cctv_future]):
                try:
                    result = future.result()
                    if future == phone_future:
                        face_signals = result
                    else:
                        body_signals = result
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
                    raise
                    
        logger.info("Completed parallel video processing")
        return face_signals, body_signals
