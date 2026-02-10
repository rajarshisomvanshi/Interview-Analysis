"""
Interview Intelligence System - Unified Timeline

Manages all timestamped events on a single unified timeline with millisecond precision.
"""

from typing import List, Optional, Type, TypeVar, Union
from collections import defaultdict
from core.schemas import (
    TimelineEvent,
    FaceSignal,
    BodySignal,
    AudioSignal,
    TranscriptSegment,
)

T = TypeVar('T', bound=TimelineEvent)


class UnifiedTimeline:
    """
    Unified timeline for managing all multimodal events.
    
    Stores and retrieves timestamped events from vision, audio, and transcript sources.
    Supports efficient range queries and temporal alignment.
    """
    
    def __init__(self):
        """Initialize empty timeline"""
        self.events: List[TimelineEvent] = []
        self._sorted = True
        
    def add_event(self, event: TimelineEvent) -> None:
        """
        Add a single event to the timeline.
        
        Args:
            event: TimelineEvent to add
        """
        self.events.append(event)
        self._sorted = False
        
    def add_events(self, events: List[TimelineEvent]) -> None:
        """
        Add multiple events to the timeline.
        
        Args:
            events: List of TimelineEvent objects to add
        """
        self.events.extend(events)
        self._sorted = False
        
    def _ensure_sorted(self) -> None:
        """Ensure events are sorted by timestamp"""
        if not self._sorted:
            self.events.sort(key=lambda e: e.timestamp_ms)
            self._sorted = True
            
    def get_events_in_range(
        self,
        start_ms: int,
        end_ms: int,
        event_type: Optional[Type[T]] = None
    ) -> List[Union[TimelineEvent, T]]:
        """
        Get all events within a time range.
        
        Args:
            start_ms: Start time in milliseconds (inclusive)
            end_ms: End time in milliseconds (inclusive)
            event_type: Optional event type to filter by (e.g., FaceSignal)
            
        Returns:
            List of events within the specified range
        """
        self._ensure_sorted()
        
        # Binary search for start position
        left, right = 0, len(self.events)
        while left < right:
            mid = (left + right) // 2
            if self.events[mid].timestamp_ms < start_ms:
                left = mid + 1
            else:
                right = mid
        start_idx = left
        
        # Collect events in range
        result = []
        for i in range(start_idx, len(self.events)):
            event = self.events[i]
            if event.timestamp_ms > end_ms:
                break
            if event_type is None or isinstance(event, event_type):
                result.append(event)
                
        return result
    
    def get_all_events(self, event_type: Optional[Type[T]] = None) -> List[Union[TimelineEvent, T]]:
        """
        Get all events, optionally filtered by type.
        
        Args:
            event_type: Optional event type to filter by
            
        Returns:
            List of all events (filtered if event_type specified)
        """
        self._ensure_sorted()
        
        if event_type is None:
            return self.events
        
        return [e for e in self.events if isinstance(e, event_type)]
    
    def get_face_signals(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[FaceSignal]:
        """Get face signals, optionally within a time range"""
        if start_ms is not None and end_ms is not None:
            return self.get_events_in_range(start_ms, end_ms, FaceSignal)
        return self.get_all_events(FaceSignal)
    
    def get_body_signals(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[BodySignal]:
        """Get body signals, optionally within a time range"""
        if start_ms is not None and end_ms is not None:
            return self.get_events_in_range(start_ms, end_ms, BodySignal)
        return self.get_all_events(BodySignal)
    
    def get_audio_signals(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[AudioSignal]:
        """Get audio signals, optionally within a time range"""
        if start_ms is not None and end_ms is not None:
            return self.get_events_in_range(start_ms, end_ms, AudioSignal)
        return self.get_all_events(AudioSignal)
    
    def get_transcript(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[TranscriptSegment]:
        """Get transcript segments, optionally within a time range"""
        if start_ms is not None and end_ms is not None:
            return self.get_events_in_range(start_ms, end_ms, TranscriptSegment)
        return self.get_all_events(TranscriptSegment)
    
    def get_duration_ms(self) -> int:
        """Get total timeline duration in milliseconds"""
        if not self.events:
            return 0
        self._ensure_sorted()
        return self.events[-1].timestamp_ms
    
    def get_event_count(self) -> int:
        """Get total number of events"""
        return len(self.events)
    
    def get_event_counts_by_type(self) -> dict:
        """Get count of events by type"""
        counts = defaultdict(int)
        for event in self.events:
            counts[event.event_type] += 1
        return dict(counts)
    
    def clear(self) -> None:
        """Clear all events from timeline"""
        self.events.clear()
        self._sorted = True
        
    def __len__(self) -> int:
        """Return number of events"""
        return len(self.events)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"UnifiedTimeline(events={len(self.events)}, duration_ms={self.get_duration_ms()})"
