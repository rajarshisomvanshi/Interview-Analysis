"""
Interview Intelligence System - API Routes

API endpoints for session management and analysis.
"""

import uuid
import asyncio
from pathlib import Path
from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Form, Header
from datetime import datetime
import logging
import traceback

from api.models import (
    SessionCreateRequest,
    SessionCreateResponse,
    AnalysisStatusResponse,
    AnalysisResultsResponse,
    TimelineDataResponse,
    ChatRequest,
    ChatResponse,
    SessionListResponse,
    SessionSummary,
    ForensicSignalsResponse,
    ForensicEventsResponse,
    ForensicScoresResponse,
    ForensicSignal,
    ForensicEvent
)
from core.schemas import SessionMetadata, SessionData, PersonIdentity, TimeSliceAnalysis, SessionAnalysis
from core.storage import storage
from core.timeline import UnifiedTimeline
from vision.pipeline import VisionPipeline
from audio.diarization import SpeakerDiarizer
from audio.transcription import Transcriber
from audio.segmentation import QuestionAnswerSegmenter
from fusion.aligner import TimelineAligner
from fusion.aggregator import SignalAggregator
from reasoning.analyzer import InterviewAnalyzer
from utils.video_utils import VideoUtils
from vision.clustering import FaceClusterer
from utils.report_generator import InterviewReportGenerator
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Debug logging setup
import os
DEBUG_LOG_FILE = "analysis_debug.log"

def log_debug(message):
    with open(DEBUG_LOG_FILE, "a") as f:
        f.write(f"{datetime.utcnow()} - {message}\n")

log_debug("API Routes module loaded")

# In-memory session status tracking (in production, use Redis or database)
session_status = {}


@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest):
    """
    Create a new interview session.
    
    Returns upload paths for videos and audio.
    """
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Create session metadata
    metadata = SessionMetadata(
        session_id=session_id,
        created_at=datetime.utcnow(),
        interviewee_name=request.interviewee_name,
        interviewee_id=request.interviewee_id,
        phone_video_path=str(settings.get_video_path(session_id, "phone")),
        cctv_video_path=str(settings.get_video_path(session_id, "cctv")),
        audio_path=str(settings.get_audio_path(session_id)),
        status="created"
    )
    
    # Create initial session data
    session_data = SessionData(metadata=metadata)
    storage.save_session_data(session_data)
    
    # Initialize status
    session_status[session_id] = {
        "status": "created",
        "progress": 0.0,
        "current_step": "Waiting for file uploads"
    }
    
    logger.info(f"Created session: {session_id}")
    
    return SessionCreateResponse(
        session_id=session_id,
        upload_urls={
            "phone_video": metadata.phone_video_path,
            "cctv_video": metadata.cctv_video_path,
            "audio": metadata.audio_path
        },
        created_at=metadata.created_at
    )


@router.post("/sessions/{session_id}/upload/phone-video")
async def upload_phone_video(session_id: str, file: UploadFile = File(...)):
    """Upload phone camera video"""
    video_path = settings.get_video_path(session_id, "phone")
    
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    logger.info(f"Uploaded phone video for session {session_id}")
    return {"status": "uploaded", "path": str(video_path)}


@router.post("/sessions/{session_id}/upload/cctv-video")
async def upload_cctv_video(session_id: str, file: UploadFile = File(...)):
    """Upload CCTV camera video"""
    video_path = settings.get_video_path(session_id, "cctv")
    
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    logger.info(f"Uploaded CCTV video for session {session_id}")
    return {"status": "uploaded", "path": str(video_path)}


@router.post("/sessions/{session_id}/upload/audio")
async def upload_audio(session_id: str, file: UploadFile = File(...)):
    """Upload audio file"""
    audio_path = settings.get_audio_path(session_id)
    
    with open(audio_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    logger.info(f"Uploaded audio for session {session_id}")
    return {"status": "uploaded", "path": str(audio_path)}


@router.post("/sessions/{session_id}/upload/single-file")
async def upload_single_file(session_id: str, file: UploadFile = File(...)):
    """
    Upload a single video file (MP4) that contains both video and audio.
    System will extract audio automatically.
    """
    # Save as phone/main video by default (can be used for both face and audio)
    # Ideally we save it as a "source" video
    source_video_path = settings.get_video_path(session_id, "source")
    
    with open(source_video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Update metadata
    session_data = storage.load_session_data(session_id)
    if session_data:
        session_data.metadata.single_video_path = str(source_video_path)
        # Also set as phone video path for vision pipeline compatibility
        session_data.metadata.phone_video_path = str(source_video_path) 
        storage.save_session_data(session_data)
        
    logger.info(f"Uploaded source video for session {session_id}")
    logger.info(f"Uploaded source video for session {session_id}")
    return {"status": "uploaded", "path": str(source_video_path)}


@router.post("/analyze-quick")
async def analyze_quick(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    interviewee_name: str = Form(...),
    user_id: str = Form(...)
):
    """
    Quick analysis endpoint: Upload video -> Create Session -> Start Analysis.
    Returns a dashboard URL for tracking.
    """
    try:
        log_debug(f"analyze_quick called for user: {user_id}")
        # 1. Create Session
        session_id = str(uuid.uuid4())
        metadata = SessionMetadata(
            session_id=session_id,
            created_at=datetime.utcnow(),
            interviewee_name=interviewee_name,
            user_id=user_id,
            status="created",
            audio_path=str(settings.get_audio_path(session_id))
        )
        
        # 2. Save Video (as single source)
        # Ensure directories exist
        settings.get_session_dir(session_id)
        
        video_path = settings.get_video_path(session_id, "source")
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Update metadata paths
        metadata.single_video_path = str(video_path)
        metadata.phone_video_path = str(video_path) # Default to phone path for pipeline
        metadata.audio_path = str(settings.get_audio_path(session_id))
        
        # Save initial data
        session_data = SessionData(metadata=metadata)
        storage.save_session_data(session_data)
        
        # Initialize status tracking
        session_status[session_id] = {
            "status": "created",
            "progress": 0.0,
            "current_step": "Video uploaded",
            "user_id": user_id
        }
        
        logger.info(f"Quick analysis started for session {session_id} (User: {user_id})")
        
        # 3. Trigger Background Analysis
        background_tasks.add_task(run_analysis, session_id)
        
        # 4. Return Dashboard URL
        return {
            "session_id": session_id,
            "status": "processing",
            "dashboard_url": f"/dashboard/{session_id}",
            "message": "Analysis started successfully"
        }
    except Exception as e:
        logger.error(f"Error in analyze_quick: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/analyze")
async def trigger_analysis(session_id: str, background_tasks: BackgroundTasks):
    """
    Trigger analysis for a session.
    
    Runs analysis in background and returns immediately.
    """
    # Check if session exists
    session_data = storage.load_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Check what files we have
    single_video_path = session_data.metadata.single_video_path
    phone_video_path = session_data.metadata.phone_video_path
    
    if single_video_path and Path(single_video_path).exists():
        # Single file flow: We need to extract audio first
        pass
    elif not phone_video_path or not Path(phone_video_path).exists():
        # Missing video (checking fallback/legacy path)
        raise HTTPException(status_code=400, detail="Missing video files")
        
    # Start analysis in background
    background_tasks.add_task(run_analysis, session_id)
    
    # Update status
    session_status[session_id] = {
        "status": "processing",
        "progress": 0.0,
        "current_step": "Starting analysis"
    }
    
    logger.info(f"Triggered analysis for session {session_id}")
    return {"status": "processing", "session_id": session_id}


def run_analysis(session_id: str):
    """
    Run complete analysis pipeline (background task).
    """
    try:
        log_debug(f"run_analysis started for session_id: {session_id}")
        logger.info(f"Starting analysis for session {session_id}")
        
        # Load session data
        session_data = storage.load_session_data(session_id)
        
        # Update status to processing immediately in storage
        session_data.metadata.status = "processing"
        storage.save_session_data(session_data)
        
        # Handle Single File / Audio Extraction
        if session_data.metadata.single_video_path:
             video_path = Path(session_data.metadata.single_video_path)
             audio_path = settings.get_audio_path(session_id)
             
             # Extract audio if it doesn't exist
             if not audio_path.exists():
                 logger.info("Extracting audio from video...")
                 session_status[session_id] = {"status": "processing", "progress": 0.05, "current_step": "Extracting audio"}
                 success = VideoUtils.extract_audio(video_path, audio_path)
                 if not success:
                     raise Exception("Audio extraction failed")
                 session_data.metadata.audio_path = str(audio_path)
                 storage.save_session_data(session_data)
        
        phone_video_path = Path(session_data.metadata.phone_video_path) if session_data.metadata.phone_video_path else None
        cctv_video_path = Path(session_data.metadata.cctv_video_path) if session_data.metadata.cctv_video_path else None
        audio_path = Path(session_data.metadata.audio_path)
        
        # Initialize timeline
        timeline = UnifiedTimeline()
        
        # Step 1: Vision processing
        session_status[session_id] = {"status": "processing", "progress": 0.1, "current_step": "Processing videos"}
        vision_pipeline = VisionPipeline()
        
        if cctv_video_path and cctv_video_path.exists():
            # Parallel processing if we have both
            face_signals, body_signals = vision_pipeline.process_videos_parallel(
                phone_video_path,
                cctv_video_path,
                timeline
            )
        else:
            # Single video flow (face only for now, unless we want to run both on same video)
            # Run face clustering here first
            session_status[session_id]["current_step"] = "Clustering faces"
            clusterer = FaceClusterer()
            identities = clusterer.process_video_for_clustering(str(phone_video_path))
            identities = clusterer.auto_assign_roles(identities)
            
            # Save identities to metadata
            converted_identities = {}
            for pid, data in identities.items():
                converted_identities[pid] = PersonIdentity(
                    id=data.id, 
                    role=data.role, 
                    num_appearances=data.num_appearances
                )
            session_data.metadata.identities = converted_identities
            storage.save_session_data(session_data)
            
            # Determine which Face Signal aligns with Interviewee
            interviewee_id = next((pid for pid, p in converted_identities.items() if p.role == 'interviewee'), None)
            session_data.metadata.interviewee_identity_id = interviewee_id
            
            # Define callback for interim results
            def save_interim_slice(slice_analysis: TimeSliceAnalysis):
                if session_data.analysis is None:
                    # Initialize empty SessionAnalysis if needed
                    session_data.analysis = SessionAnalysis(
                        session_id=session_id,
                        analyzed_at=datetime.now(),
                        integrity_score=0.0,
                        confidence_score=0.0,
                        risk_score=0.0
                    )
                
                # Append slice
                session_data.analysis.slices.append(slice_analysis)
                storage.save_session_data(session_data)
                
                # Update in-memory status so polling gets it
                if session_id in session_status:
                    current_slices = session_status[session_id].get("slices", [])
                    # Remove potential duplicate by start_ms
                    current_slices = [s for s in current_slices if s.get('start_ms') != slice_analysis.start_ms]
                    current_slices.append(slice_analysis.model_dump())
                    session_status[session_id]["slices"] = current_slices
                    
                    # Update progress: Step 1 is vision (10% to 40%)
                    if total_duration > 0:
                        progress = 0.1 + (0.3 * (slice_analysis.end_ms / (total_duration * 1000)))
                        session_status[session_id]["progress"] = min(0.4, progress)
                    
                logger.info(f"Saved interim slice {len(session_data.analysis.slices)} to session data. Progress: {session_status[session_id].get('progress')}")

            # Now process video for signals
            session_status[session_id]["current_step"] = "Processing video"
            logger.info(f"Processing phone video for session {session_id}")
            
            face_signals = vision_pipeline.process_phone_video(
                phone_video_path,
                timeline,
                save_callback=save_interim_slice
            )
            
            # Since we don't have CCTV, body signals might be limited or derived from phone if possible
            # For now, empty list or maybe basic body analysis from phone video if implemented
            body_signals = [] 
            
            # Add signals to timeline
            for signal in face_signals:
                timeline.add_event(signal)


        
        # Step 2: Audio processing
        session_status[session_id] = {"status": "processing", "progress": 0.4, "current_step": "Processing audio"}
        
        # Diarization
        diarizer = SpeakerDiarizer()
        speaker_segments = diarizer.diarize(audio_path)
        labeled_segments = diarizer.label_speakers(speaker_segments)
        
        # Transcription
        transcriber = Transcriber()
        transcription_result = transcriber.transcribe(audio_path)
        words = transcription_result["words"]
        transcript_segments = transcriber.align_with_diarization(words, labeled_segments)
        
        # Add to timeline
        for segment in transcript_segments:
            timeline.add_event(segment)
            
        # Step 3: Question-answer segmentation
        session_status[session_id] = {"status": "processing", "progress": 0.6, "current_step": "Segmenting Q&A pairs"}
        segmenter = QuestionAnswerSegmenter()
        qa_pairs = segmenter.segment(transcript_segments)
        
        # Step 4: Signal aggregation
        session_status[session_id] = {"status": "processing", "progress": 0.7, "current_step": "Aggregating signals"}
        aggregator = SignalAggregator()
        qa_pairs = aggregator.aggregate_for_qa_pairs(qa_pairs, timeline, str(audio_path))
        
        # Step 5: LLM analysis
        session_status[session_id] = {"status": "processing", "progress": 0.8, "current_step": "Performing LLM analysis"}
        analyzer = InterviewAnalyzer()
        
        # Define callback for slice progress
        def update_progress(current, total, current_slices):
            # Scale progress from 0.8 to 0.9
            progress = 0.8 + (0.1 * (current / total))
            status_update = {
                "status": "processing",
                "progress": progress,
                "current_step": f"Analyzing time slice {current}/{total}",
                "slices": current_slices 
            }
            session_status[session_id].update(status_update)
            
        # Analyze time slices explicitly
        slices_data = []
        if transcript_segments:
             session_status[session_id]["current_step"] = "Starting time slice analysis"
             slices_data = analyzer.analyze_time_slices(
                 transcript_segments, 
                 timeline.get_duration_ms(), 
                 60000, # 1 minute slices
                 progress_callback=update_progress
             )
        
        analysis = analyzer.analyze_session(
            session_id, 
            qa_pairs, 
            timeline.get_duration_ms(),
            transcript_segments=transcript_segments,
            slices=slices_data
        )
        
        # Step 6: Save results
        session_status[session_id] = {"status": "processing", "progress": 0.9, "current_step": "Saving results"}
        session_data.question_answer_pairs = qa_pairs
        session_data.analysis = analysis
        if hasattr(analysis, 'slices') and analysis.slices:
             # Ensure slices in session_status are fully updated with final ones
             session_status[session_id]["slices"] = [s.model_dump() for s in analysis.slices]
             
        session_data.metadata.status = "completed"
        
        storage.save_session_data(session_data)
        storage.save_analysis(session_id, analysis)
        
        # Step 7: Generate Markdown Report
        session_status[session_id]["current_step"] = "Generating report"
        report_gen = InterviewReportGenerator(session_data)
        report_path = settings.get_session_dir(session_id) / "analysis_report.md"
        report_gen.save_report(report_path)
        
        # Complete
        final_slices = [s.model_dump() for s in analysis.slices] if hasattr(analysis, 'slices') and analysis.slices else []
        session_status[session_id] = {
            "status": "completed",
            "progress": 1.0,
            "current_step": "Analysis complete",
            "slices": final_slices
        }
        logger.info(f"Analysis complete for session {session_id}")
        
    except Exception as e:
        logger.error(f"Analysis failed for session {session_id}: {e}")
        session_status[session_id] = {
            "status": "failed",
            "progress": 0.0,
            "current_step": "Analysis failed",
            "error_message": str(e)
        }
        log_debug(f"run_analysis failed for {session_id}: {e}")
        import traceback
        log_debug(traceback.format_exc())


@router.get("/sessions/{session_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(session_id: str):
    """Get analysis status for a session"""
    if session_id not in session_status:
        print(f"DEBUG: Session {session_id} not in memory")
        # Check if status exists in storage (for restored or interrupted sessions)
        try:
            session_data = storage.load_session_data(session_id)
            print(f"DEBUG: load_session_data returned: {session_data is not None}")
        except Exception as e:
            print(f"DEBUG: load_session_data failed: {e}")
            session_data = None
            
        if session_data:
            # If it was processing but not in memory, it implies interruption
            status = session_data.metadata.status
            error_message = None
            
            if status == "processing":
                status = "failed"
                error_message = "Analysis interrupted by server restart"
                
            return AnalysisStatusResponse(
                session_id=session_id,
                status=status,
                progress=1.0 if status == "completed" else 0.0,
                current_step="Restored from storage",
                error_message=error_message,
                slices=[s.model_dump() for s in session_data.analysis.slices] if session_data.analysis and session_data.analysis.slices else []
            )
        raise HTTPException(status_code=404, detail=f"Session not found. Debug: load_session_data returned None. Path: {storage.data_dir}")
        
    status_info = session_status[session_id]
    
    return AnalysisStatusResponse(
        session_id=session_id,
        status=status_info["status"],
        progress=status_info.get("progress"),
        current_step=status_info.get("current_step"),
        error_message=status_info.get("error_message"),
        slices=status_info.get("slices")
    )


@router.get("/sessions/{session_id}/results", response_model=AnalysisResultsResponse)
async def get_analysis_results(session_id: str):
    """Get analysis results for a session"""
    # Load analysis
    analysis = storage.load_analysis(session_id)
    
    # Check if session data has interim analysis
    session_data = storage.load_session_data(session_id)
    
    if not analysis and session_data and session_data.analysis:
        # Use interim analysis from session_data
        analysis = session_data.analysis
    
    if not analysis:
        if session_data and session_data.metadata.status == "failed":
             raise HTTPException(status_code=400, detail="Analysis failed for this session")
        # If processing, maybe return partial? 
        # But allow 404 if truly nothing is there
        raise HTTPException(status_code=404, detail="Analysis results not found")
        
    # Load session data for metadata
    session_data = storage.load_session_data(session_id)
    
    return AnalysisResultsResponse(
        session_id=session_id,
        analyzed_at=analysis.analyzed_at,
        question_count=len(analysis.question_analyses),
        session_duration_ms=session_data.metadata.duration_ms or 0,
        executive_summary=analysis.executive_summary,
        download_url=f"/sessions/{session_id}/download/analysis",
        slices=[
            TimeSliceResponse(
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                insight=s.insight,
                score=s.score,
                summary=s.summary,
                scene_description=s.scene_description,
                dialogue=s.dialogue,
                behavioral_analysis=s.behavioral_analysis
            ) for s in analysis.slices
        ] if hasattr(analysis, "slices") and analysis.slices else []
    )


@router.get("/sessions/{session_id}/download/analysis-report")
async def download_analysis_report(session_id: str):
    """Download the generated Markdown report"""
    report_path = settings.get_session_dir(session_id) / "analysis_report.md"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found. Has the analysis completed?")
        
    from fastapi.responses import FileResponse
    return FileResponse(
        path=report_path,
        filename=f"interview_analysis_{session_id}.md",
        media_type="text/markdown"
    )


@router.get("/videos/{session_id}/{filename}")
async def serve_video(session_id: str, filename: str, range_header: Optional[str] = Header(None, alias="Range")):
    """Serve video with partial content (Range) support for seeking"""
    video_path = settings.get_session_dir(session_id) / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
        
    file_size = video_path.stat().st_size
    
    if range_header:
        start, end = 0, file_size - 1
        range_str = range_header.replace("bytes=", "")
        parts = range_str.split("-")
        
        if parts[0]:
            start = int(parts[0])
        if len(parts) > 1 and parts[1]:
            end = int(parts[1])
            
        # Ensure bounds
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        
        chunk_size = (end - start) + 1
        
        def iter_file():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    chunk = f.read(min(remaining, 1024 * 1024)) # 1MB chunks
                    if not chunk:
                        break
                    yield chunk
                    remaining -= len(chunk)
                    
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            iter_file(),
            status_code=206,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(chunk_size),
                "Content-Type": "video/mp4",
            }
        )
    
    from fastapi.responses import FileResponse
    return FileResponse(video_path, media_type="video/mp4")


@router.get("/sessions/{session_id}/timeline", response_model=TimelineDataResponse)
async def get_timeline_data(session_id: str):
    """Get timeline data for a session"""
    session_data = storage.load_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Reconstruct timeline from session data
    timeline = UnifiedTimeline()
    if session_data.face_signals:
        timeline.add_events(session_data.face_signals)
    if session_data.body_signals:
        timeline.add_events(session_data.body_signals)
    if session_data.audio_signals:
        timeline.add_events(session_data.audio_signals)
        
    return TimelineDataResponse(
        session_id=session_id,
        duration_ms=timeline.get_duration_ms(),
        event_counts=timeline.get_event_counts_by_type(),
        download_url=f"/sessions/{session_id}/download/timeline"
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, keep_analysis: bool = False):
    """Delete a session"""
    success = storage.delete_session(session_id, keep_analysis=keep_analysis)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Remove from status tracking
    if session_id in session_status:
        del session_status[session_id]
        
    return {"status": "deleted", "session_id": session_id}


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """List all interview sessions"""
    try:
        session_ids = storage.list_sessions()
        summaries = []
        
        for sid in session_ids:
            # Load metadata
            # We need to implement get_session_metadata in storage or load full data
            metadata = storage.get_session_metadata(sid)
            if not metadata:
                continue

            # Use in-memory status if available (more up-to-date)
            current_status = metadata.status
            if sid in session_status:
                current_status = session_status[sid]["status"]
            elif current_status == "processing":
                 # If it says processing but not in memory, it was interrupted
                 current_status = "failed"
                
            # Try to load analysis for scores
            analysis = storage.load_analysis(sid)
            
            summary = SessionSummary(
                session_id=sid,
                interviewee_name=metadata.interviewee_name,
                created_at=metadata.created_at,
                status=current_status,
                executive_summary=analysis.executive_summary if analysis else None,
                integrity_score=analysis.integrity_score if analysis and hasattr(analysis, 'integrity_score') else None,
                confidence_score=analysis.confidence_score if analysis and hasattr(analysis, 'confidence_score') else None,
                risk_score=analysis.risk_score if analysis and hasattr(analysis, 'risk_score') else None
            )
            summaries.append(summary)
            
        # Sort by date desc
        summaries.sort(key=lambda x: x.created_at, reverse=True)
        
        return SessionListResponse(sessions=summaries)
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat_with_session(session_id: str, request: ChatRequest):
    """Chat with the AI about a specific session"""
    try:
        # 1. Try to load final analysis results
        analysis = storage.load_analysis(session_id)
        
        # 2. Fallback to interim analysis in session data if final isn't ready
        if not analysis:
            session_data = storage.load_session_data(session_id)
            if session_data and session_data.analysis:
                analysis = session_data.analysis
        
        if not analysis or (not analysis.slices and not analysis.executive_summary):
            # Check if session exists at all
            if not storage.load_session_data(session_id):
                raise HTTPException(status_code=404, detail="Session not found")
            raise HTTPException(status_code=400, detail="Analysis hasn't started yet. Please wait for the first results to appear.")
            
        # Initialize analyzer
        analyzer = InterviewAnalyzer()
        
        # Generate response
        response_text = analyzer.chat_with_context(analysis, request.message, request.history)
        
        return ChatResponse(response=response_text)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- New Forensic Modular Endpoints ---

@router.get("/sessions/{session_id}/signals", response_model=ForensicSignalsResponse)
async def get_forensic_signals(session_id: str, signal_type: Optional[str] = None):
    """Get granular forensic signal vectors for a session"""
    session_data = storage.load_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
        
    signals = []
    
    # Process Face Signals
    if not signal_type or signal_type == "face":
        for s in session_data.face_signals:
            if s.landmarks:
                signals.append(ForensicSignal(
                    timestamp_ms=s.timestamp_ms,
                    signal_type="face_mesh",
                    values={"landmarks_count": float(len(s.landmarks))}, # Simplified for now
                    confidence=s.confidence
                ))
            if s.emotions:
                signals.append(ForensicSignal(
                    timestamp_ms=s.timestamp_ms,
                    signal_type="face_emotions",
                    values=s.emotions,
                    confidence=s.confidence
                ))

    # Process Audio Signals
    if not signal_type or signal_type == "audio":
        for s in session_data.audio_signals:
            signals.append(ForensicSignal(
                timestamp_ms=s.timestamp_ms,
                signal_type="audio_telemetry",
                values={
                    "pitch_hz": s.pitch_hz or 0.0,
                    "energy_db": s.energy_db or 0.0,
                    "speech_rate_wpm": s.speech_rate_wpm or 0.0
                },
                confidence=s.confidence
            ))

    return ForensicSignalsResponse(session_id=session_id, signals=signals)


@router.get("/sessions/{session_id}/events", response_model=ForensicEventsResponse)
async def get_forensic_events(session_id: str):
    """Get semantic forensic events detected in the session"""
    session_data = storage.load_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
        
    events = []
    
    # Extract posture shifts from body signals
    for s in session_data.body_signals:
        if s.posture_shift_detected:
            events.append(ForensicEvent(
                start_ms=s.timestamp_ms,
                end_ms=s.timestamp_ms + (s.duration_ms or 1000),
                event_type="postural_shift",
                description="Significant change in candidate posture detected.",
                severity=0.7,
                metadata={"torso_angle": str(s.torso_angle)}
            ))
            
    # Extract pauses from audio
    for s in session_data.audio_signals:
        if s.pause_detected:
            events.append(ForensicEvent(
                start_ms=s.timestamp_ms,
                end_ms=s.timestamp_ms + (s.pause_duration_ms or 500),
                event_type="hesitation_pause",
                description="Long silent pause detected in response.",
                severity=0.5,
                metadata={"duration_ms": str(s.pause_duration_ms)}
            ))

    return ForensicEventsResponse(session_id=session_id, events=events)


@router.get("/sessions/{session_id}/scores/forensic", response_model=ForensicScoresResponse)
async def get_forensic_scores(session_id: str):
    """Get weighted forensic scores for a session"""
    session_data = storage.load_session_data(session_id)
    if not session_data or not session_data.analysis:
        raise HTTPException(status_code=404, detail="Analysis results not found")
        
    analysis = session_data.analysis
    
    scores = {
        "integrity_score": analysis.integrity_score,
        "confidence_score": analysis.confidence_score,
        "risk_score": analysis.risk_score
    }
    
    # Default mock weights (in future, these can be dynamic)
    weights = {
        "visual_weight": 0.4,
        "vocal_weight": 0.3,
        "linguistic_weight": 0.3
    }
    
    return ForensicScoresResponse(
        session_id=session_id,
        scores=scores,
        weights_used=weights,
        recalculated_at=datetime.utcnow()
    )


@router.post("/sessions/{session_id}/recalculate")
async def recalculate_forensic_scores(session_id: str, weights: Optional[Dict[str, float]] = None):
    """Recalculate forensic scores with new weights (UI Placeholder)"""
    # Logic to trigger score aggregation logic with custom weights
    # For now, just acknowledged
    return {"status": "success", "message": "Recalculation logic triggered (Placeholder)"}
