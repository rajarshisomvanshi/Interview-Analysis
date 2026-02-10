"""
Interview Intelligence System - API Models

Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from core.schemas import QuestionAnswerPair


class SessionCreateRequest(BaseModel):
    """Request to create a new interview session"""
    interviewee_name: Optional[str] = Field(None, description="Interviewee name (optional)")
    interviewee_id: Optional[str] = Field(None, description="Interviewee ID (optional)")


class SessionCreateResponse(BaseModel):
    """Response after creating a session"""
    session_id: str = Field(..., description="Unique session identifier")
    upload_urls: dict = Field(..., description="URLs/paths for uploading videos and audio")
    created_at: datetime = Field(..., description="Session creation timestamp")


class AnalysisStatusResponse(BaseModel):
    """Response for analysis status check"""
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Status: created, processing, completed, failed")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress percentage (0-1)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    estimated_completion_ms: Optional[int] = Field(None, description="Estimated time to completion")
    slices: Optional[List[Dict]] = Field(None, description="List of completed time slices")

class TimeSliceResponse(BaseModel):
    """Response for a time slice analysis"""
    start_ms: int
    end_ms: int
    insight: str
    score: float
    summary: str
    scene_description: Optional[str] = None
    dialogue: Optional[str] = None
    behavioral_analysis: Optional[str] = None


class AnalysisResultsResponse(BaseModel):
    """Response containing analysis results"""
    session_id: str = Field(..., description="Session identifier")
    analyzed_at: datetime = Field(..., description="Analysis completion timestamp")
    question_count: int = Field(..., description="Number of questions analyzed")
    session_duration_ms: int = Field(..., description="Total session duration")
    executive_summary: str
    integrity_score: Optional[float] = Field(None, description="0-100 Integrity Score")
    confidence_score: Optional[float] = Field(None, description="0-100 Confidence Score")
    risk_score: Optional[float] = Field(None, description="0-100 Risk Score")
    download_url: Optional[str] = Field(None, description="URL to download full results JSON")
    slices: Optional[List[TimeSliceResponse]] = Field(None, description="List of time slices")
    qa_pairs: Optional[List[QuestionAnswerPair]] = Field(None, description="List of Q&A pairs associated with the session")


class TimelineDataResponse(BaseModel):
    """Response containing timeline data"""
    session_id: str = Field(..., description="Session identifier")
    duration_ms: int = Field(..., description="Timeline duration")
    event_counts: dict = Field(..., description="Count of events by type")
    download_url: Optional[str] = Field(None, description="URL to download full timeline JSON")


class ChatRequest(BaseModel):
    """Request for chatbot interaction"""
    message: str = Field(..., description="User message")
    history: List[dict] = Field(default_factory=list, description="Chat history [{'role': 'user', 'content': '...'}, ...]")


class ChatResponse(BaseModel):
    """Response from chatbot"""
    response: str = Field(..., description="LLM response")


class TranslationRequest(BaseModel):
    """Request to translate text"""
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(default="Hindi", description="Target language")
    
class TranslationResponse(BaseModel):
    """Response from translation"""
    translated_text: str = Field(..., description="Translated text")


class SessionSummary(BaseModel):
    """Summary of a session for listing"""
    session_id: str
    interviewee_name: Optional[str]
    created_at: datetime
    status: str
    executive_summary: Optional[str] = None
    integrity_score: Optional[float] = None
    confidence_score: Optional[float] = None
    risk_score: Optional[float] = None


class SessionListResponse(BaseModel):
    """List of sessions"""
    sessions: List[SessionSummary]


# --- New Forensic Modular Models ---

class ForensicSignal(BaseModel):
    """Raw signal vector at a specific timestamp"""
    timestamp_ms: int
    signal_type: str  # e.g., "face_mesh", "audio_pitch", "pose_vector"
    values: Dict[str, float]
    confidence: float

class ForensicEvent(BaseModel):
    """Semantic event detected in the interview"""
    start_ms: int
    end_ms: int
    event_type: str  # e.g., "postural_collapse", "micro_expression", "stress_spike"
    description: str
    severity: float  # 0-1
    metadata: Dict[str, str] = Field(default_factory=dict)

class ForensicSignalsResponse(BaseModel):
    """Response containing granular signal vectors"""
    session_id: str
    signals: List[ForensicSignal]

class ForensicEventsResponse(BaseModel):
    """Response containing semantic forensic events"""
    session_id: str
    events: List[ForensicEvent]

class ForensicScoresResponse(BaseModel):
    """Response containing weighted forensic scores"""
    session_id: str
    scores: Dict[str, float]  # e.g., {"integrity": 85.0, "stress_index": 42.0}
    weights_used: Dict[str, float]
    recalculated_at: datetime
