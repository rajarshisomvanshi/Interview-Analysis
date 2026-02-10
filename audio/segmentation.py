"""
Interview Intelligence System - Question-Answer Segmentation

Segments transcript into question-answer pairs.
"""

import re
from typing import List, Tuple, Optional
import logging

from core.schemas import TranscriptSegment, SpeakerLabel, QuestionAnswerPair
from config import settings

logger = logging.getLogger(__name__)


class QuestionAnswerSegmenter:
    """
    Segments interview transcript into question-answer pairs.
    """
    
    # Question markers (words that often indicate questions)
    QUESTION_WORDS = {
        'what', 'when', 'where', 'who', 'whom', 'whose', 'why', 'which', 'how',
        'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might',
        'do', 'does', 'did', 'is', 'are', 'was', 'were', 'have', 'has', 'had'
    }
    
    def __init__(self, min_silence_gap_ms: int = 500):
        """
        Initialize segmenter.
        
        Args:
            min_silence_gap_ms: Minimum silence gap to consider as boundary
        """
        self.min_silence_gap_ms = min_silence_gap_ms
        logger.info("Initialized QuestionAnswerSegmenter")
        
    def segment(
        self,
        transcript_segments: List[TranscriptSegment]
    ) -> List[QuestionAnswerPair]:
        """
        Segment transcript into question-answer pairs.
        
        Args:
            transcript_segments: List of TranscriptSegment objects
            
        Returns:
            List of QuestionAnswerPair objects
        """
        logger.info(f"Segmenting {len(transcript_segments)} transcript segments")
        
        # Group segments by speaker turns
        speaker_turns = self._group_by_speaker_turns(transcript_segments)
        
        # Identify question-answer pairs
        qa_pairs = self._identify_qa_pairs(speaker_turns)
        
        logger.info(f"Identified {len(qa_pairs)} question-answer pairs")
        return qa_pairs
        
    def _group_by_speaker_turns(
        self,
        transcript_segments: List[TranscriptSegment]
    ) -> List[Tuple[SpeakerLabel, List[TranscriptSegment]]]:
        """
        Group consecutive segments by speaker.
        
        Returns:
            List of (speaker, segments) tuples
        """
        if not transcript_segments:
            return []
            
        turns = []
        current_speaker = transcript_segments[0].speaker
        current_segments = [transcript_segments[0]]
        
        for segment in transcript_segments[1:]:
            # Check for speaker change or silence gap
            time_gap = segment.timestamp_ms - current_segments[-1].end_ms
            
            if segment.speaker != current_speaker or time_gap > self.min_silence_gap_ms:
                # Save current turn
                turns.append((current_speaker, current_segments))
                
                # Start new turn
                current_speaker = segment.speaker
                current_segments = [segment]
            else:
                # Continue current turn
                current_segments.append(segment)
                
        # Add final turn
        if current_segments:
            turns.append((current_speaker, current_segments))
            
        return turns
        
    def _identify_qa_pairs(
        self,
        speaker_turns: List[Tuple[SpeakerLabel, List[TranscriptSegment]]]
    ) -> List[QuestionAnswerPair]:
        """
        Identify question-answer pairs from speaker turns.
        Supports bidirectional questioning (standard and reverse).
        """
        qa_pairs = []
        qa_index = 0
        
        for i in range(len(speaker_turns) - 1):
            speaker, segments = speaker_turns[i]
            next_speaker, next_segments = speaker_turns[i + 1]
            
            text = ' '.join([s.text for s in segments])
            
            # Check if this turn is a question
            if self._is_question(text):
                # We have a question, the next turn is likely the answer
                # Regardless of who is the speaker
                question_text = text
                question_start_ms = segments[0].timestamp_ms
                question_end_ms = segments[-1].timestamp_ms + (segments[-1].duration_ms or 0)
                
                response_text = ' '.join([s.text for s in next_segments])
                response_start_ms = next_segments[0].timestamp_ms
                response_end_ms = next_segments[-1].timestamp_ms + (next_segments[-1].duration_ms or 0)
                
                # Create QA pair
                qa_pair = QuestionAnswerPair(
                    qa_index=qa_index,
                    question_text=f"[{speaker.value.upper()}] {question_text}",
                    question_start_ms=question_start_ms,
                    question_end_ms=question_end_ms,
                    response_text=f"[{next_speaker.value.upper()}] {response_text}",
                    response_start_ms=response_start_ms,
                    response_end_ms=response_end_ms,
                    response_latency_ms=max(0, response_start_ms - question_end_ms)
                )
                
                # Add filler word info for interviewee responses specifically
                if next_speaker == SpeakerLabel.INTERVIEWEE:
                    filler_count, fillers = self.count_filler_words(response_text)
                    qa_pair.filler_word_count = filler_count
                    qa_pair.filler_words = fillers

                qa_pairs.append(qa_pair)
                qa_index += 1
                
        return qa_pairs
        
    def _is_question(self, text: str) -> bool:
        """
        Heuristic to determine if text is a question.
        
        Args:
            text: Text to check
            
        Returns:
            True if likely a question
        """
        text_lower = text.lower().strip()
        
        # Check for question mark
        if text.endswith('?'):
            return True
            
        # Check for question words at start
        words = text_lower.split()
        if words and words[0] in self.QUESTION_WORDS:
            return True
            
        return False
        
    def count_filler_words(self, text: str) -> Tuple[int, List[str]]:
        """
        Count filler words in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (count, list of filler words found)
        """
        filler_patterns = [
            r'\bum\b', r'\buh\b', r'\blike\b', r'\byou know\b',
            r'\bbasically\b', r'\bactually\b', r'\bso\b', r'\bwell\b',
            r'\bI mean\b', r'\bright\b', r'\bokay\b'
        ]
        
        filler_words = []
        text_lower = text.lower()
        
        for pattern in filler_patterns:
            matches = re.findall(pattern, text_lower)
            filler_words.extend(matches)
            
        return len(filler_words), filler_words
