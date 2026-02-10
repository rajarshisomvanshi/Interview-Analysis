"""
Interview Intelligence System - Configuration Management

Centralized configuration using Pydantic settings with environment variable support.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, DirectoryPath
from pathlib import Path
from typing import Literal, Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # ========================================================================
    # Model Paths and Configuration
    # ========================================================================
    
    # YOLO models
    yolo_phone_model: str = Field(
        default="yolov8n.pt",
        description="YOLO model for phone video (face detection)"
    )
    yolo_cctv_model: str = Field(
        default="yolov8s-pose.pt",
        description="YOLO model for CCTV video (pose estimation)"
    )
    
    # Face analysis
    face_recognition_backend: Literal["facenet", "arcface", "deepface"] = Field(
        default="facenet",
        description="Face recognition model backend"
    )
    face_mesh_backend: Literal["mediapipe", "openface"] = Field(
        default="mediapipe",
        description="Facial landmark extraction backend"
    )
    use_emotion_classifier: bool = Field(
        default=True,
        description="Whether to use emotion classifier as weak prior"
    )
    emotion_model_path: Optional[str] = Field(
        default=None,
        description="Path to emotion classifier model (optional)"
    )
    
    # Audio models
    whisper_model: str = Field(
        default="base",
        description="Whisper model size (tiny, base, small, medium, large)"
    )
    use_faster_whisper: bool = Field(
        default=True,
        description="Use faster-whisper for optimized inference"
    )
    diarization_model: str = Field(
        default="pyannote/speaker-diarization",
        description="Speaker diarization model"
    )
    
    # ========================================================================
    # Processing Parameters
    # ========================================================================
    
    # Video processing
    video_fps: int = Field(
        default=30,
        ge=1,
        le=60,
        description="Target FPS for video processing"
    )
    phone_video_resolution: tuple[int, int] = Field(
        default=(1920, 1080),
        description="Phone video resolution (width, height)"
    )
    cctv_video_resolution: tuple[int, int] = Field(
        default=(1280, 720),
        description="CCTV video resolution (width, height)"
    )
    
    # Audio processing
    audio_sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz"
    )
    
    # Detection thresholds
    face_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for face detection"
    )
    body_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for body keypoint detection"
    )
    face_recognition_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for face recognition match"
    )
    
    # Signal extraction windows
    blink_detection_window_ms: int = Field(
        default=200,
        description="Time window for blink detection in milliseconds"
    )
    movement_analysis_window_ms: int = Field(
        default=1000,
        description="Time window for movement analysis in milliseconds"
    )
    
    # ========================================================================
    # Storage Configuration
    # ========================================================================
    
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root directory for data storage"
    )
    video_retention_days: int = Field(
        default=7,
        ge=0,
        description="Number of days to retain raw video files (0 = delete immediately)"
    )
    enable_video_storage: bool = Field(
        default=True,
        description="Whether to store raw video files"
    )
    
    # ========================================================================
    # LLM Configuration
    # ========================================================================
    
    llm_provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="ollama",
        description="LLM provider for analysis"
    )
    llm_model: str = Field(
        default="qwen2.5:1.5b",
        description="LLM model name"
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation"
    )
    llm_max_tokens: int = Field(
        default=1000,
        ge=100,
        description="Maximum tokens for LLM response"
    )
    
    # API keys (loaded from environment)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_base_url: Optional[str] = Field(
        default=None,
        description="OpenAI base URL (for compatible APIs)"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    
    # Ollama configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )

    # Local LLM configuration (Transformers)
    use_local_llm: bool = Field(
        default=True,
        description="Use local transformers model instead of API"
    )
    local_llm_model: str = Field(
        default="qwen2.5:1.5b",
        description="Ollama model tag for local LLM"
    )
    
    # ========================================================================
    # API Configuration
    # ========================================================================
    
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port"
    )
    api_workers: int = Field(
        default=1,
        ge=1,
        description="Number of API workers"
    )
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS for API"
    )
    
    # ========================================================================
    # Performance and Edge Optimization
    # ========================================================================
    
    use_gpu: bool = Field(
        default=True,
        description="Use GPU acceleration if available"
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for model inference"
    )
    num_workers: int = Field(
        default=4,
        ge=1,
        description="Number of worker threads for parallel processing"
    )
    enable_model_quantization: bool = Field(
        default=False,
        description="Enable model quantization for edge deployment"
    )
    
    # ========================================================================
    # Logging and Debugging
    # ========================================================================
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    save_debug_frames: bool = Field(
        default=False,
        description="Save debug frames with visualizations"
    )
    debug_output_dir: Path = Field(
        default=Path("./debug"),
        description="Directory for debug outputs"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
        
    def get_session_dir(self, session_id: str) -> Path:
        """Get directory path for a specific session"""
        session_dir = self.data_dir / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def get_video_path(self, session_id: str, video_type: Literal["phone", "cctv"]) -> Path:
        """Get video file path for a session"""
        return self.get_session_dir(session_id) / f"{video_type}_video.mp4"
    
    def get_audio_path(self, session_id: str) -> Path:
        """Get audio file path for a session"""
        return self.get_session_dir(session_id) / "audio.wav"
    
    def get_session_data_path(self, session_id: str) -> Path:
        """Get session data JSON file path"""
        return self.get_session_dir(session_id) / "session_data.json"
    
    def get_analysis_path(self, session_id: str) -> Path:
        """Get analysis results JSON file path"""
        return self.get_session_dir(session_id) / "analysis_results.json"


# Global settings instance
settings = Settings()
