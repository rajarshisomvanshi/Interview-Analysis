"""
Interview Intelligence System - Core Data Schemas

Defines all Pydantic models for type safety and validation across the system.
"""

from pydantic import BaseModel, Field, validator, ConfigDict
from typing import List, Dict, Optional, Literal
from datetime import datetime
from uuid import UUID, uuid4
import enum

# Enums
class SpeakerLabel(str, enum.Enum):
    INTERVIEWER = "interviewer"
    INTERVIEWEE = "interviewee"
    UNKNOWN = "unknown"

class SignalType(str, enum.Enum):
    FACE = "face"
    BODY = "body"
    AUDIO = "audio"

# Base Timeline Event
class TimelineEvent(BaseModel):
    """Base class for all events on the unified timeline"""
    model_config = ConfigDict(validate_assignment=True)
    
    timestamp_ms: int = Field(..., description="Timestamp in milliseconds from start")
    duration_ms: Optional[int] = Field(None, description="Duration of event if applicable")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score of detection")

# Vision Signals
class FaceSignal(TimelineEvent):
    """Facial behavior signals at a specific timestamp"""
    face_detected: bool = True
    identity_id: Optional[str] = Field(None, description="ID of the person (e.g., person_0)")
    identity_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence of identity match")
    box: Optional[List[float]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    
    # Gaze & Eyes
    eye_contact: Optional[float] = Field(None, ge=0.0, le=1.0, description="1.0 = direct eye contact")
    blink_detected: bool = False
    left_eye_open: Optional[float] = Field(None, ge=0.0, le=1.0)
    right_eye_open: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Emotional/Behavioral Indicators (Action Units & Weak Priors)
    action_units: Dict[str, float] = Field(default_factory=dict, description="AU intensities (e.g. AU4_brow_lowerer)")
    emotions: Dict[str, float] = Field(default_factory=dict, description="Weak priors from emotion classifier (e.g. 'happy': 0.1)")
    head_pose: Optional[Dict[str, float]] = Field(None, description="yaw, pitch, roll")
    gaze_direction: Optional[Dict[str, float]] = Field(None, description="Gaze vector (x, y, z)")
    landmarks: Optional[List[Dict[str, float]]] = Field(None, description="Facial landmarks [x, y, z]")

class BodySignal(TimelineEvent):
    """Body movement signals at a specific timestamp"""
    body_detected: bool = True
    keypoints: Optional[List[List[float]]] = Field(None, description="Pose keypoints")
    
    # Movement Indicators
    hand_movement_intensity: Optional[float] = Field(None, ge=0.0, description="Intensity of hand/arm movement")
    posture_shift_detected: bool = False
    torso_angle: Optional[float] = Field(None, description="Leaning angle")
    leg_movement_intensity: Optional[float] = Field(None, ge=0.0, description="Intensity of leg/foot movement")

# Audio Signals
class AudioSignal(TimelineEvent):
    """Audio-derived signals"""
    speech_rate_wpm: Optional[float] = None
    pitch_hz: Optional[float] = None
    pitch_std: Optional[float] = None
    energy_db: Optional[float] = None
    pause_detected: bool = False
    pause_duration_ms: Optional[int] = None

class TranscriptSegment(TimelineEvent):
    """Segment of transcribed text"""
    text: str
    speaker: SpeakerLabel = SpeakerLabel.UNKNOWN
    speaker_id: Optional[str] = None # 'SPEAKER_00', etc.
    words: List[Dict] = Field(default_factory=list, description="Word-level timestamps")
    end_ms: int

# Aggregated Data Structures
class QuestionAnswerPair(BaseModel):
    """Aligned Q&A pair with all associated signals"""
    qa_index: int
    question_text: str
    response_text: str
    
    # Timing
    question_start_ms: int
    question_end_ms: int
    response_start_ms: int
    response_end_ms: int
    response_latency_ms: int
    
    # Aggregated Signals (Summary Statistics)
    face_signals_summary: Dict[str, float] = Field(default_factory=dict)
    body_signals_summary: Dict[str, float] = Field(default_factory=dict)
    audio_signals_summary: Dict[str, float] = Field(default_factory=dict)
    
    # Specific metrics
    filler_word_count: int = 0
    filler_words: List[str] = Field(default_factory=list)

class PersonIdentity(BaseModel):
    """Identity information for a detected person"""
    id: str = Field(..., description="Unique ID (person_0)")
    role: str = Field("unknown", description="interviewee, interviewer_1, etc.")
    name: Optional[str] = None
    num_appearances: int = 0
    thumbnail_path: Optional[str] = None # Path to saved face image

class SessionMetadata(BaseModel):
    """Metadata for an interview session"""
    session_id: str
    created_at: datetime
    user_id: Optional[str] = None # Added for Quick API
    interviewee_name: Optional[str] = None
    interviewee_id: Optional[str] = None
    status: str = "created" # created, processing, completed, failed
    
    # File Paths
    phone_video_path: Optional[str] = None # Can be None if single video used
    cctv_video_path: Optional[str] = None
    single_video_path: Optional[str] = None # NEW: For single stream multi-POV
    audio_path: str
    
    # Duration
    duration_ms: Optional[int] = None
    
    # Identities
    identities: Dict[str, PersonIdentity] = Field(default_factory=dict)
    interviewee_identity_id: Optional[str] = None # Which person_id is the interviewee

class QuestionAnalysis(BaseModel):
    """LLM analysis of a single question"""
    qa_index: int
    communication_clarity: str
    confidence_indicators: str
    stress_indicators: str
    summary: str
    analysis_confidence: Literal["low", "medium", "high"]

class TimeSliceAnalysis(BaseModel):
    """Analysis of a specific time segment"""
    start_ms: int
    end_ms: int
    insight: str
    score: float = Field(0.0, ge=0, le=100, description="Slice-specific specific score")
    summary: str
    fluency: Optional[float] = None
    confidence: Optional[float] = None
    attitude: Optional[float] = None
    
    # Detailed LLM Output
    scene_description: Optional[str] = None
    dialogue: Optional[str] = None
    behavioral_analysis: Optional[str] = None

class SessionAnalysis(BaseModel):
    """Full session analysis"""
    session_id: str
    analyzed_at: datetime
    question_analyses: List[QuestionAnalysis] = Field(default_factory=list)
    overall_trends: str = ""
    communication_patterns: str = ""
    behavioral_patterns: str = ""
    executive_summary: str = ""
    
    # Quantitative Scoring
    integrity_score: float = Field(0.0, ge=0, le=100, description="0-100 score indicating likelihood of honesty")
    confidence_score: float = Field(0.0, ge=0, le=100, description="0-100 score indicating displayed confidence")
    risk_score: float = Field(0.0, ge=0, le=100, description="0-100 score indicating potential risk/deception")
    
    # Time Slices
    slices: List[TimeSliceAnalysis] = Field(default_factory=list, description="Time-segmented analysis (e.g. 2-min chunks)")

class SessionData(BaseModel):
    """Complete container for a session"""
    metadata: SessionMetadata
    timeline_events: List[TimelineEvent] = Field(default_factory=list) # Raw events
    face_signals: List[FaceSignal] = Field(default_factory=list)
    body_signals: List[BodySignal] = Field(default_factory=list)
    audio_signals: List[AudioSignal] = Field(default_factory=list)
    transcript_segments: List[TranscriptSegment] = Field(default_factory=list)
    question_answer_pairs: List[QuestionAnswerPair] = Field(default_factory=list)
    analysis: Optional[SessionAnalysis] = None
