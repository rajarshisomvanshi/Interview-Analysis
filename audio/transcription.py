"""
Interview Intelligence System - Speech-to-Text Transcription

Performs speech-to-text with word-level timestamps using Whisper.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    try:
        import whisper
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        logging.warning("Neither faster-whisper nor whisper installed")

from core.schemas import TranscriptSegment, SpeakerLabel
from audio.diarization import SpeakerSegment
from config import settings

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Speech-to-text transcription with word-level timestamps.
    """
    
    def __init__(self, model_size: str = None):
        """
        Initialize transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        model_size = model_size or settings.whisper_model
        
        if settings.use_faster_whisper and FASTER_WHISPER_AVAILABLE:
            # Use faster-whisper for optimized inference
            device = "cuda" if settings.use_gpu else "cpu"
            compute_type = "float16" if settings.use_gpu else "int8"
            
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            self.use_faster_whisper = True
            logger.info(f"Initialized Transcriber with faster-whisper ({model_size}, {device})")
            
        elif WHISPER_AVAILABLE:
            # Use standard whisper
            device = "cuda" if settings.use_gpu else "cpu"
            self.model = whisper.load_model(model_size, device=device)
            self.use_faster_whisper = False
            logger.info(f"Initialized Transcriber with whisper ({model_size}, {device})")
            
        else:
            raise ImportError("Neither faster-whisper nor whisper available")
            
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: None for auto-detect)
            
        Returns:
            Dictionary with 'words' and 'language'
        """
        logger.info(f"Transcribing audio: {audio_path}")
        
        if self.use_faster_whisper:
            words, detected_lang = self._transcribe_faster_whisper(audio_path, language)
        else:
            words, detected_lang = self._transcribe_whisper(audio_path, language)
            
        return {
            "words": words,
            "language": detected_lang
        }
            
    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        language: Optional[str]
    ) -> Tuple[List[Dict], str]:
        """Transcribe using faster-whisper"""
        # Optimized prompt for Hinglish (Hindi + English) in interviews
        initial_prompt = "This is an interview conversation in Hinglish, mixing Hindi and English languages."
        
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            beam_size=5,
            vad_filter=True,

        )
        
        detected_lang = info.language
        logger.info(f"Detected language: {detected_lang} (probability: {info.language_probability})")
        
        words = []
        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    words.append({
                        'word': word.word.strip(),
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    })
                    
        logger.info(f"Transcribed {len(words)} words")
        return words, detected_lang
        
    def _transcribe_whisper(
        self,
        audio_path: Path,
        language: Optional[str]
    ) -> Tuple[List[Dict], str]:
        """Transcribe using standard whisper"""
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True
        )
        
        detected_lang = result.get('language', 'en')
        logger.info(f"Detected language (standard whisper): {detected_lang}")
        
        words = []
        for segment in result.get('segments', []):
            if 'words' in segment:
                for word_info in segment['words']:
                    words.append({
                        'word': word_info['word'].strip(),
                        'start': word_info['start'],
                        'end': word_info['end'],
                        'probability': word_info.get('probability', 1.0)
                    })
                    
        logger.info(f"Transcribed {len(words)} words")
        return words, detected_lang
        
    def align_with_diarization(
        self,
        words: List[Dict],
        speaker_segments: List[tuple]
    ) -> List[TranscriptSegment]:
        """
        Align transcribed words with speaker diarization.
        
        Args:
            words: List of word dictionaries from transcription
            speaker_segments: List of (SpeakerSegment, SpeakerLabel) tuples
            
        Returns:
            List of TranscriptSegment objects with speaker labels
        """
        transcript_segments = []
        
        for word_info in words:
            word_start_ms = int(word_info['start'] * 1000)
            word_end_ms = int(word_info['end'] * 1000)
            word_mid_ms = (word_start_ms + word_end_ms) // 2
            
            # Find speaker for this word (based on midpoint)
            speaker = SpeakerLabel.UNKNOWN
            for segment, label in speaker_segments:
                if segment.start_ms <= word_mid_ms <= segment.end_ms:
                    speaker = label
                    break
                    
            # Create transcript segment
            transcript_segment = TranscriptSegment(
                timestamp_ms=word_start_ms,
                text=word_info['word'],  # Add text field
                word=word_info['word'],
                start_ms=word_start_ms,
                end_ms=word_end_ms,
                confidence=word_info.get('probability', 1.0),
                speaker=speaker
            )
            
            transcript_segments.append(transcript_segment)
            
        logger.info(f"Created {len(transcript_segments)} transcript segments")
        return transcript_segments
        
    def get_full_transcript(
        self,
        transcript_segments: List[TranscriptSegment],
        speaker_filter: Optional[SpeakerLabel] = None
    ) -> str:
        """
        Get full transcript text, optionally filtered by speaker.
        
        Args:
            transcript_segments: List of TranscriptSegment objects
            speaker_filter: Optional speaker to filter by
            
        Returns:
            Full transcript text
        """
        words = []
        for segment in transcript_segments:
            if speaker_filter is None or segment.speaker == speaker_filter:
                words.append(segment.word)
                
        return ' '.join(words)
