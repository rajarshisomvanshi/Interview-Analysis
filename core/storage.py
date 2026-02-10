"""
Interview Intelligence System - Storage Layer

Handles persistent storage of session data with retention policies.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

from core.schemas import SessionData, SessionMetadata, SessionAnalysis
from config import settings

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages storage and retrieval of interview session data.
    
    Features:
    - JSON storage for structured signals and analysis (permanent)
    - Optional video storage with configurable retention period
    - Automatic cleanup of expired video files
    """
    
    def __init__(self):
        """Initialize storage manager"""
        self.data_dir = settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def save_session_data(self, session_data: SessionData) -> Path:
        """
        Save complete session data to JSON.
        
        Args:
            session_data: SessionData object to save
            
        Returns:
            Path to saved JSON file
        """
        session_id = session_data.metadata.session_id
        output_path = settings.get_session_data_path(session_id)
        
        # Convert to JSON
        data_dict = session_data.model_dump(mode='json')
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved session data to {output_path}")
        return output_path
    
    def load_session_data(self, session_id: str) -> Optional[SessionData]:
        """
        Load session data from JSON.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionData object or None if not found
        """
        data_path = settings.get_session_data_path(session_id)
        
        if not data_path.exists():
            logger.warning(f"Session data not found: {session_id}")
            return None
            
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            return SessionData(**data_dict)
        except json.JSONDecodeError:
            logger.error(f"Corrupt session data JSON: {session_id}")
            return None
        except Exception as e:
            logger.error(f"Error loading session data {session_id}: {e}")
            return None
    
    def save_analysis(self, session_id: str, analysis: SessionAnalysis) -> Path:
        """
        Save analysis results to JSON.
        
        Args:
            session_id: Session identifier
            analysis: SessionAnalysis object
            
        Returns:
            Path to saved JSON file
        """
        output_path = settings.get_analysis_path(session_id)
        
        # Convert to JSON
        analysis_dict = analysis.model_dump(mode='json')
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved analysis to {output_path}")
        return output_path
    
    def load_analysis(self, session_id: str) -> Optional[SessionAnalysis]:
        """
        Load analysis results from JSON.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionAnalysis object or None if not found
        """
        analysis_path = settings.get_analysis_path(session_id)
        
        if not analysis_path.exists():
            logger.warning(f"Analysis not found: {session_id}")
            return None
            
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_dict = json.load(f)
            return SessionAnalysis(**analysis_dict)
        except json.JSONDecodeError:
            logger.error(f"Corrupt analysis JSON: {session_id}")
            return None
        except Exception as e:
            logger.error(f"Error loading analysis {session_id}: {e}")
            return None
    
    def save_video(self, session_id: str, video_type: str, source_path: Path) -> Path:
        """
        Save video file with optional retention policy.
        
        Args:
            session_id: Session identifier
            video_type: 'phone' or 'cctv'
            source_path: Path to source video file
            
        Returns:
            Path to saved video file
        """
        if not settings.enable_video_storage:
            logger.info("Video storage disabled, skipping")
            return source_path
            
        dest_path = settings.get_video_path(session_id, video_type)
        
        # Copy video file
        shutil.copy2(source_path, dest_path)
        logger.info(f"Saved {video_type} video to {dest_path}")
        
        return dest_path
    
    def save_audio(self, session_id: str, source_path: Path) -> Path:
        """
        Save audio file.
        
        Args:
            session_id: Session identifier
            source_path: Path to source audio file
            
        Returns:
            Path to saved audio file
        """
        dest_path = settings.get_audio_path(session_id)
        
        # Copy audio file
        shutil.copy2(source_path, dest_path)
        logger.info(f"Saved audio to {dest_path}")
        
        return dest_path
    
    def cleanup_expired_videos(self) -> int:
        """
        Delete video files older than retention period.
        
        Returns:
            Number of files deleted
        """
        if settings.video_retention_days == 0:
            logger.info("Video retention is 0 days, no cleanup needed")
            return 0
            
        sessions_dir = self.data_dir / "sessions"
        if not sessions_dir.exists():
            return 0
            
        cutoff_time = datetime.utcnow() - timedelta(days=settings.video_retention_days)
        deleted_count = 0
        
        for session_dir in sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
                
            # Check session metadata for creation time
            metadata_path = session_dir / "session_data.json"
            if not metadata_path.exists():
                continue
                
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    created_at = datetime.fromisoformat(data['metadata']['created_at'].replace('Z', '+00:00'))
                    
                if created_at < cutoff_time:
                    # Delete video files only, keep JSON data
                    for video_file in session_dir.glob("*.mp4"):
                        video_file.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted expired video: {video_file}")
                        
                    for audio_file in session_dir.glob("*.wav"):
                        audio_file.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted expired audio: {audio_file}")
                        
            except Exception as e:
                logger.error(f"Error processing session {session_dir.name}: {e}")
                
        logger.info(f"Cleanup complete: deleted {deleted_count} files")
        return deleted_count
    
    def delete_session(self, session_id: str, keep_analysis: bool = False) -> bool:
        """
        Delete all data for a session.
        
        Args:
            session_id: Session identifier
            keep_analysis: If True, keep analysis JSON but delete everything else
            
        Returns:
            True if deletion successful
        """
        session_dir = settings.get_session_dir(session_id)
        
        if not session_dir.exists():
            logger.warning(f"Session directory not found: {session_id}")
            return False
            
        if keep_analysis:
            # Delete everything except analysis
            analysis_path = settings.get_analysis_path(session_id)
            for item in session_dir.iterdir():
                if item != analysis_path:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            logger.info(f"Deleted session data (kept analysis): {session_id}")
        else:
            # Delete entire session directory
            shutil.rmtree(session_dir)
            logger.info(f"Deleted session: {session_id}")
            
        return True
    
    def list_sessions(self) -> list[str]:
        """
        List all session IDs.
        
        Returns:
            List of session IDs
        """
        sessions_dir = self.data_dir / "sessions"
        if not sessions_dir.exists():
            return []
            
        return [d.name for d in sessions_dir.iterdir() if d.is_dir()]
    
    def get_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """
        Get session metadata without loading full session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionMetadata or None if not found
        """
        data_path = settings.get_session_data_path(session_id)
        
        if not data_path.exists():
            return None
            
        with open(data_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            
        return SessionMetadata(**data_dict['metadata'])


# Global storage manager instance
storage = StorageManager()
