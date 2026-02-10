"""
Interview Intelligence System - Video Utilities

Utilities for video processing and audio extraction.
"""

import cv2
import subprocess
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)

class VideoUtils:
    """
    Utilities for handling video files.
    """
    
    @staticmethod
    def extract_audio(video_path: Path, output_audio_path: Path) -> bool:
        """
        Extract audio from video file using robust strategy.
        Tries:
        1. imageio-ffmpeg (Guaranteed binary)
        2. System ffmpeg (Fallback)
        """
        try:
            # Strategy 1: imageio-ffmpeg (Best for portability)
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            cmd = [
                ffmpeg_exe, "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(output_audio_path)
            ]
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Extraction successful via imageio-ffmpeg")
                return True
            else:
                logger.warning(f"imageio-ffmpeg extraction failed with code {result.returncode}: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"imageio-ffmpeg extraction exception: {e}")
            
        # Strategy 2: System ffmpeg (Fallback)
        try:
            if shutil.which("ffmpeg"):
                logger.info("Attempting fallback to system ffmpeg")
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    str(output_audio_path)
                ]
                logger.info(f"Running fallback command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return True
                else:
                    logger.error(f"System ffmpeg fallback failed with code {result.returncode}: {result.stderr}")
        except Exception as e:
            logger.error(f"System ffmpeg fallback exception: {e}")

        logger.error("All audio extraction methods failed.")
        return False

    @staticmethod
    def get_video_duration_ms(video_path: Path) -> int:
        """Get video duration in milliseconds using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_ms = int((frame_count / fps) * 1000) if fps > 0 else 0
    @staticmethod
    def extract_audio_segment(video_path: Path, output_audio_path: Path, start_ms: int, end_ms: int) -> bool:
        """
        Extract specific audio segment from video.
        
        Args:
            video_path: Source video path
            output_audio_path: Destination audio path (.wav)
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
        """
        duration_sec = (end_ms - start_ms) / 1000.0
        start_sec = start_ms / 1000.0
        
        # Use system ffmpeg generally available
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(video_path),
            "-t", f"{duration_sec:.3f}",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_audio_path)
        ]
        
        try:
            # logger.info(f"Extracting audio segment: {start_sec}s - {start_sec+duration_sec}s")
            # Suppress output unless error
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return True
            else:
                logger.warning(f"Audio segment extraction failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Audio segment extraction exception: {e}")
            return False
