"""
Interview Intelligence System - Speaker Diarization

Separates and labels speakers (interviewer vs interviewee).
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("pyannote.audio not installed. Install with: pip install pyannote.audio")

from core.schemas import SpeakerLabel
from config import settings

logger = logging.getLogger(__name__)


class SpeakerSegment:
    """Represents a speaker segment with timestamps"""
    
    def __init__(self, start_ms: int, end_ms: int, speaker_id: str):
        """
        Initialize speaker segment.
        
        Args:
            start_ms: Segment start time in milliseconds
            end_ms: Segment end time in milliseconds
            speaker_id: Speaker identifier
        """
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.speaker_id = speaker_id
        
    def __repr__(self) -> str:
        return f"SpeakerSegment({self.start_ms}-{self.end_ms}ms, speaker={self.speaker_id})"


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio.
    
    Separates audio into speaker segments and labels them as interviewer/interviewee.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize speaker diarizer.
        
        Args:
            model_name: Pyannote model name (default: from settings)
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError("pyannote.audio required. Install with: pip install pyannote.audio")
            
        model_name = model_name or settings.diarization_model
        
        # Note: Requires HuggingFace token for pyannote models
        # Set HF_TOKEN environment variable or pass use_auth_token parameter
        try:
            self.pipeline = Pipeline.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Could not load pyannote model: {e}")
            logger.warning("Using fallback simple diarization")
            self.pipeline = None
            
        logger.info(f"Initialized SpeakerDiarizer")
        
    def diarize(self, audio_path: Path) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of SpeakerSegment objects
        """
        if self.pipeline is None:
            return self._fallback_diarization(audio_path)
            
        logger.info(f"Performing diarization on: {audio_path}")
        
        # Run diarization
        diarization = self.pipeline(str(audio_path))
        
        # Convert to SpeakerSegment objects
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                start_ms=int(turn.start * 1000),
                end_ms=int(turn.end * 1000),
                speaker_id=speaker
            )
            segments.append(segment)
            
        logger.info(f"Found {len(segments)} speaker segments")
        return segments
        
    def _fallback_diarization(self, audio_path: Path) -> List[SpeakerSegment]:
        """
        Fallback diarization using simple voice activity detection.
        
        This is a placeholder - in production, use proper diarization.
        """
        logger.warning("Using fallback diarization (not production-ready)")
        
        # Placeholder: create dummy segments
        # In production, implement VAD-based segmentation
        segments = [
            SpeakerSegment(0, 5000, "SPEAKER_00"),
            SpeakerSegment(5000, 10000, "SPEAKER_01"),
        ]
        
        return segments
        
    def label_speakers(
        self,
        segments: List[SpeakerSegment],
        interviewer_speaker_id: str = None
    ) -> List[Tuple[SpeakerSegment, SpeakerLabel]]:
        """
        Label speakers as interviewer or interviewee.
        
        Args:
            segments: List of speaker segments
            interviewer_speaker_id: Known interviewer speaker ID (optional)
            
        Returns:
            List of (segment, label) tuples
        """
        if not segments:
            return []
            
        # Count speaker occurrences
        speaker_counts = {}
        for segment in segments:
            speaker_counts[segment.speaker_id] = speaker_counts.get(segment.speaker_id, 0) + 1
            
        # If interviewer ID provided, use it
        if interviewer_speaker_id:
            labeled_segments = []
            for segment in segments:
                if segment.speaker_id == interviewer_speaker_id:
                    label = SpeakerLabel.INTERVIEWER
                else:
                    label = SpeakerLabel.INTERVIEWEE
                labeled_segments.append((segment, label))
            return labeled_segments
            
        # Otherwise, assume speaker with fewer turns is interviewer
        # (interviewee typically speaks more in responses)
        sorted_speakers = sorted(speaker_counts.items(), key=lambda x: x[1])
        
        if len(sorted_speakers) >= 2:
            interviewer_id = sorted_speakers[0][0]
            interviewee_id = sorted_speakers[1][0]
        elif len(sorted_speakers) == 1:
            # Only one speaker detected - label as interviewee
            interviewer_id = None
            interviewee_id = sorted_speakers[0][0]
        else:
            return []
            
        # Label segments
        labeled_segments = []
        for segment in segments:
            if segment.speaker_id == interviewer_id:
                label = SpeakerLabel.INTERVIEWER
            elif segment.speaker_id == interviewee_id:
                label = SpeakerLabel.INTERVIEWEE
            else:
                label = SpeakerLabel.UNKNOWN
            labeled_segments.append((segment, label))
            
        logger.info(f"Labeled {len(labeled_segments)} segments")
        return labeled_segments
