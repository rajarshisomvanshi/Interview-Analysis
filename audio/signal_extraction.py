"""
Interview Intelligence System - Audio Signal Extraction

Extracts behavioral signals from audio (speech rate, pitch, fluency, etc.).
"""

import numpy as np
from typing import List, Dict, Optional
import logging

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not installed. Install with: pip install librosa")

from core.schemas import TranscriptSegment, AudioSignal, SpeakerLabel
from config import settings

logger = logging.getLogger(__name__)


class AudioSignalExtractor:
    """
    Extract audio-derived behavioral signals.
    """
    
    def __init__(self):
        """Initialize audio signal extractor"""
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required. Install with: pip install librosa")
            
        logger.info("Initialized AudioSignalExtractor")
        
    def extract_signals_from_audio(
        self,
        audio_path: str,
        start_ms: int,
        end_ms: int
    ) -> Dict[str, float]:
        """
        Extract audio signals from a specific time range.
        
        Args:
            audio_path: Path to audio file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            
        Returns:
            Dictionary of audio signal metrics
        """
        # Load audio segment
        y, sr = librosa.load(
            audio_path,
            sr=settings.audio_sample_rate,
            offset=start_ms / 1000.0,
            duration=(end_ms - start_ms) / 1000.0
        )
        
        signals = {}
        
        # Pitch (F0) extraction
        pitch, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Filter out unvoiced frames
        pitch_voiced = pitch[voiced_flag]
        
        if len(pitch_voiced) > 0:
            signals['pitch_hz'] = float(np.nanmean(pitch_voiced))
            signals['pitch_std'] = float(np.nanstd(pitch_voiced))
        else:
            signals['pitch_hz'] = 0.0
            signals['pitch_std'] = 0.0
            
        # Energy/volume
        rms = librosa.feature.rms(y=y)[0]
        signals['energy_db'] = float(librosa.amplitude_to_db(np.mean(rms)))
        
        return signals
        
    def compute_speech_rate(
        self,
        transcript_segments: List[TranscriptSegment],
        start_ms: int,
        end_ms: int
    ) -> float:
        """
        Compute speech rate (words per minute) for a time range.
        
        Args:
            transcript_segments: List of transcript segments
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            
        Returns:
            Speech rate in words per minute
        """
        # Count words in range
        word_count = 0
        for segment in transcript_segments:
            if start_ms <= segment.timestamp_ms < end_ms:
                word_count += 1
                
        # Calculate duration in minutes
        duration_min = (end_ms - start_ms) / 60000.0
        
        if duration_min > 0:
            speech_rate = word_count / duration_min
        else:
            speech_rate = 0.0
            
        return speech_rate
        
    def detect_pauses(
        self,
        transcript_segments: List[TranscriptSegment],
        min_pause_ms: int = 200
    ) -> List[Dict]:
        """
        Detect pauses in speech.
        
        Args:
            transcript_segments: List of transcript segments
            min_pause_ms: Minimum pause duration to detect
            
        Returns:
            List of pause dictionaries with start, end, duration
        """
        pauses = []
        
        for i in range(len(transcript_segments) - 1):
            current_end = transcript_segments[i].end_ms
            next_start = transcript_segments[i + 1].timestamp_ms
            
            pause_duration = next_start - current_end
            
            if pause_duration >= min_pause_ms:
                pauses.append({
                    'start_ms': current_end,
                    'end_ms': next_start,
                    'duration_ms': pause_duration
                })
                
        return pauses
        
    def create_audio_signals(
        self,
        audio_path: str,
        transcript_segments: List[TranscriptSegment],
        window_ms: int = 1000
    ) -> List[AudioSignal]:
        """
        Create AudioSignal objects for the entire audio.
        
        Args:
            audio_path: Path to audio file
            transcript_segments: List of transcript segments
            window_ms: Window size for signal extraction
            
        Returns:
            List of AudioSignal objects
        """
        if not transcript_segments:
            return []
            
        audio_signals = []
        
        # Get total duration
        start_ms = transcript_segments[0].timestamp_ms
        end_ms = transcript_segments[-1].end_ms
        
        # Extract signals in windows
        current_ms = start_ms
        
        while current_ms < end_ms:
            window_end_ms = min(current_ms + window_ms, end_ms)
            
            # Extract audio features
            try:
                audio_features = self.extract_signals_from_audio(
                    audio_path,
                    current_ms,
                    window_end_ms
                )
            except Exception as e:
                logger.warning(f"Failed to extract audio features at {current_ms}ms: {e}")
                audio_features = {}
                
            # Compute speech rate
            speech_rate = self.compute_speech_rate(
                transcript_segments,
                current_ms,
                window_end_ms
            )
            
            # Detect pauses in window
            window_segments = [
                s for s in transcript_segments
                if current_ms <= s.timestamp_ms < window_end_ms
            ]
            pauses = self.detect_pauses(window_segments)
            
            # Create signal
            signal = AudioSignal(
                timestamp_ms=current_ms,
                speech_rate_wpm=speech_rate,
                pitch_hz=audio_features.get('pitch_hz'),
                pitch_std=audio_features.get('pitch_std'),
                energy_db=audio_features.get('energy_db'),
                pause_detected=len(pauses) > 0,
                pause_duration_ms=pauses[0]['duration_ms'] if pauses else None
            )
            
            audio_signals.append(signal)
            current_ms += window_ms
            
        logger.info(f"Created {len(audio_signals)} audio signals")
        return audio_signals


class ResponseMetricsCalculator:
    """
    Calculate response-specific metrics (latency, fluency, etc.).
    """
    
    def __init__(self):
        """Initialize response metrics calculator"""
        logger.info("Initialized ResponseMetricsCalculator")
        
    def calculate_response_metrics(
        self,
        response_segments: List[TranscriptSegment],
        audio_path: str
    ) -> Dict[str, any]:
        """
        Calculate metrics for a response.
        
        Args:
            response_segments: Transcript segments for the response
            audio_path: Path to audio file
            
        Returns:
            Dictionary of response metrics
        """
        if not response_segments:
            return {}
            
        metrics = {}
        
        # Speech rate
        start_ms = response_segments[0].timestamp_ms
        end_ms = response_segments[-1].end_ms
        duration_min = (end_ms - start_ms) / 60000.0
        
        if duration_min > 0:
            metrics['speech_rate_wpm'] = len(response_segments) / duration_min
        else:
            metrics['speech_rate_wpm'] = 0.0
            
        # Pause analysis
        extractor = AudioSignalExtractor()
        pauses = extractor.detect_pauses(response_segments)
        
        metrics['pause_count'] = len(pauses)
        if pauses:
            metrics['avg_pause_duration_ms'] = np.mean([p['duration_ms'] for p in pauses])
            metrics['total_pause_duration_ms'] = sum(p['duration_ms'] for p in pauses)
        else:
            metrics['avg_pause_duration_ms'] = 0.0
            metrics['total_pause_duration_ms'] = 0.0
            
        # Fluency score (inverse of pause ratio)
        total_duration_ms = end_ms - start_ms
        if total_duration_ms > 0:
            pause_ratio = metrics['total_pause_duration_ms'] / total_duration_ms
            metrics['fluency_score'] = 1.0 - pause_ratio
        else:
            metrics['fluency_score'] = 1.0
            
        # Pitch stability
        try:
            audio_features = extractor.extract_signals_from_audio(
                audio_path,
                start_ms,
                end_ms
            )
            metrics['pitch_mean_hz'] = audio_features.get('pitch_hz', 0.0)
            metrics['pitch_stability'] = 1.0 / (1.0 + audio_features.get('pitch_std', 0.0))
        except Exception as e:
            logger.warning(f"Failed to extract pitch features: {e}")
            metrics['pitch_mean_hz'] = 0.0
            metrics['pitch_stability'] = 0.0
            
        return metrics
