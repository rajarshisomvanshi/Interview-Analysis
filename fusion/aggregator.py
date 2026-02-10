"""
Interview Intelligence System - Signal Aggregation

Aggregates multimodal signals per question-answer pair.
"""

import numpy as np
from typing import List, Dict
import logging

from core.schemas import (
    FaceSignal,
    BodySignal,
    AudioSignal,
    TranscriptSegment,
    QuestionAnswerPair,
    SpeakerLabel
)
from core.timeline import UnifiedTimeline
from audio.segmentation import QuestionAnswerSegmenter
from audio.signal_extraction import ResponseMetricsCalculator

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    Aggregates multimodal signals for question-answer pairs.
    """
    
    def __init__(self):
        """Initialize signal aggregator"""
        self.metrics_calculator = ResponseMetricsCalculator()
        logger.info("Initialized SignalAggregator")
        
    def aggregate_for_qa_pairs(
        self,
        qa_pairs: List[QuestionAnswerPair],
        timeline: UnifiedTimeline,
        audio_path: str
    ) -> List[QuestionAnswerPair]:
        """
        Aggregate signals for each question-answer pair.
        
        Args:
            qa_pairs: List of QuestionAnswerPair objects
            timeline: UnifiedTimeline with all events
            audio_path: Path to audio file
            
        Returns:
            List of QuestionAnswerPair objects with aggregated signals
        """
        logger.info(f"Aggregating signals for {len(qa_pairs)} Q&A pairs")
        
        for qa_pair in qa_pairs:
            # Aggregate face signals during response
            face_signals = timeline.get_face_signals(
                qa_pair.response_start_ms,
                qa_pair.response_end_ms
            )
            qa_pair.face_signals_summary = self._aggregate_face_signals(face_signals)
            
            # Aggregate body signals during response
            body_signals = timeline.get_body_signals(
                qa_pair.response_start_ms,
                qa_pair.response_end_ms
            )
            qa_pair.body_signals_summary = self._aggregate_body_signals(body_signals)
            
            # Aggregate audio signals during response
            audio_signals = timeline.get_audio_signals(
                qa_pair.response_start_ms,
                qa_pair.response_end_ms
            )
            qa_pair.audio_signals_summary = self._aggregate_audio_signals(audio_signals)
            
            # Get transcript segments for response
            transcript = timeline.get_transcript(
                qa_pair.response_start_ms,
                qa_pair.response_end_ms
            )
            
            # Calculate response metrics
            response_metrics = self.metrics_calculator.calculate_response_metrics(
                transcript,
                audio_path
            )
            qa_pair.audio_signals_summary.update(response_metrics)
            
            # Count filler words
            segmenter = QuestionAnswerSegmenter()
            filler_count, filler_words = segmenter.count_filler_words(qa_pair.response_text)
            qa_pair.filler_word_count = filler_count
            qa_pair.filler_words = filler_words
            
        logger.info("Signal aggregation complete")
        return qa_pairs
        
    def _aggregate_face_signals(self, face_signals: List[FaceSignal]) -> Dict[str, float]:
        """
        Aggregate face signals into summary statistics.
        
        Args:
            face_signals: List of FaceSignal objects
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not face_signals:
            return {}
            
        summary = {}
        
        # Eye contact
        eye_contact_values = [s.eye_contact for s in face_signals if s.eye_contact is not None]
        if eye_contact_values:
            summary['eye_contact_mean'] = float(np.mean(eye_contact_values))
            summary['eye_contact_std'] = float(np.std(eye_contact_values))
            summary['eye_contact_min'] = float(np.min(eye_contact_values))
            summary['eye_contact_max'] = float(np.max(eye_contact_values))
            
        # Blink rate
        blinks = [s for s in face_signals if s.blink_detected]
        duration_min = (face_signals[-1].timestamp_ms - face_signals[0].timestamp_ms) / 60000.0
        if duration_min > 0:
            summary['blink_rate_per_min'] = len(blinks) / duration_min
        else:
            summary['blink_rate_per_min'] = 0.0
            
        # Eye openness
        left_eye_values = [s.left_eye_open for s in face_signals if s.left_eye_open is not None]
        right_eye_values = [s.right_eye_open for s in face_signals if s.right_eye_open is not None]
        
        if left_eye_values and right_eye_values:
            summary['eye_openness_mean'] = float(np.mean(left_eye_values + right_eye_values))
            
        # Action Units (average intensities)
        au_values = {}
        for signal in face_signals:
            if signal.action_units:
                for au, intensity in signal.action_units.items():
                    if au not in au_values:
                        au_values[au] = []
                    au_values[au].append(intensity)
                    
        for au, values in au_values.items():
            summary[f'{au}_mean'] = float(np.mean(values))
            summary[f'{au}_std'] = float(np.std(values))
            
        return summary
        
    def _aggregate_body_signals(self, body_signals: List[BodySignal]) -> Dict[str, float]:
        """
        Aggregate body signals into summary statistics.
        
        Args:
            body_signals: List of BodySignal objects
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not body_signals:
            return {}
            
        summary = {}
        
        # Hand movement
        hand_movement_values = [
            s.hand_movement_intensity for s in body_signals
            if s.hand_movement_intensity is not None
        ]
        if hand_movement_values:
            summary['hand_movement_mean'] = float(np.mean(hand_movement_values))
            summary['hand_movement_std'] = float(np.std(hand_movement_values))
            summary['hand_movement_max'] = float(np.max(hand_movement_values))
            
        # Posture shifts
        posture_shifts = [s for s in body_signals if s.posture_shift_detected]
        summary['posture_shift_count'] = len(posture_shifts)
        
        # Torso angle
        torso_angles = [s.torso_angle for s in body_signals if s.torso_angle is not None]
        if torso_angles:
            summary['torso_angle_mean'] = float(np.mean(torso_angles))
            summary['torso_angle_std'] = float(np.std(torso_angles))
            
        # Leg movement
        leg_movement_values = [
            s.leg_movement_intensity for s in body_signals
            if s.leg_movement_intensity is not None
        ]
        if leg_movement_values:
            summary['leg_movement_mean'] = float(np.mean(leg_movement_values))
            summary['leg_movement_std'] = float(np.std(leg_movement_values))
            
        return summary
        
    def _aggregate_audio_signals(self, audio_signals: List[AudioSignal]) -> Dict[str, float]:
        """
        Aggregate audio signals into summary statistics.
        
        Args:
            audio_signals: List of AudioSignal objects
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not audio_signals:
            return {}
            
        summary = {}
        
        # Speech rate
        speech_rates = [s.speech_rate_wpm for s in audio_signals if s.speech_rate_wpm is not None]
        if speech_rates:
            summary['speech_rate_mean'] = float(np.mean(speech_rates))
            summary['speech_rate_std'] = float(np.std(speech_rates))
            
        # Pitch
        pitch_values = [s.pitch_hz for s in audio_signals if s.pitch_hz is not None]
        if pitch_values:
            summary['pitch_mean'] = float(np.mean(pitch_values))
            summary['pitch_std'] = float(np.std(pitch_values))
            
        # Pitch stability
        pitch_std_values = [s.pitch_std for s in audio_signals if s.pitch_std is not None]
        if pitch_std_values:
            summary['pitch_stability_mean'] = float(np.mean(pitch_std_values))
            
        # Energy
        energy_values = [s.energy_db for s in audio_signals if s.energy_db is not None]
        if energy_values:
            summary['energy_mean'] = float(np.mean(energy_values))
            summary['energy_std'] = float(np.std(energy_values))
            
        # Pauses
        pauses = [s for s in audio_signals if s.pause_detected]
        summary['pause_count'] = len(pauses)
        
        return summary


# Fusion package initialization
__all__ = ["SignalAggregator"]
