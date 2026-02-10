import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.getcwd())

from core.storage import StorageManager
from reasoning.analyzer import InterviewAnalyzer
from core.schemas import SessionAnalysis

def regenerate(session_id):
    print(f"Regenerating summary for session {session_id}...")
    storage = StorageManager()
    
    # Load data
    session_data = storage.load_session_data(session_id)
    if not session_data:
        print("SessionData not found.")
        return

    print(f"Loaded session with {len(session_data.question_answer_pairs)} Q&A pairs.")
    
    # Initialize Analyzer
    analyzer = InterviewAnalyzer()
    
    # Re-run session analysis (this will trigger the new prompt)
    # We pass existing data to avoid re-transcription/segmentation
    try:
        # Note: analyze_session expects duration in ms
        duration = session_data.metadata.duration_ms or 0
        
        # We need to extract the 'slices' dicts from existing analysis if possible, 
        # or let analyzer re-do it? 
        # analyzer.analyze_session calls analyze_time_slices if transcript is present.
        # But we want to keep existing slices if they are good, or just refresh the summary.
        # The prompt fix is in analyze_session -> _call_llm.
        
        # Let's just run it. It might re-do slices but that's fast if using transcript.
        # Wait, analyze_session does NOT re-do slices if we pass them?
        # Line 217 in analyzer.py: if transcript_segments: ... analyze_time_slices...
        
        # To avoid overwriting valid slices with potential errors, let's just focus on the summary part.
        # But analyze_session is monolithic.
        # Let's trust it.
        
        # Prepare context
        slices_data = [] # Let it re-generate or specific logic?
        # Actually, let's pass existing slices to avoid re-analysis cost if possible?
        # analyzer.py line 215 prioritization: if transcript_segments, it re-runs slices.
        
        # Serialize transcript segments to dicts as analyzer expects
        transcript_dicts = [s.model_dump() for s in session_data.transcript_segments]
        
        # Fallback: Synthesize transcript from Q&A if missing
        if not transcript_dicts and session_data.question_answer_pairs:
            print("Synthesizing transcript from Q&A pairs...")
            for qa in session_data.question_answer_pairs:
                # Question
                transcript_dicts.append({
                    "text": qa.question_text,
                    "speaker": "interviewer",
                    "timestamp_ms": qa.question_start_ms,
                    "end_ms": qa.question_end_ms
                })
                # Answer
                transcript_dicts.append({
                    "text": qa.response_text,
                    "speaker": "interviewee",
                    "timestamp_ms": qa.response_start_ms,
                    "end_ms": qa.response_end_ms
                })
        
        print("Running analyze_session...")
        new_analysis = analyzer.analyze_session(
            session_id=session_id,
            qa_pairs=session_data.question_answer_pairs,
            session_duration_ms=duration,
            transcript_segments=transcript_dicts
        )
        
        print("Analysis complete.")
        print(f"New Summary: {new_analysis.executive_summary[:100]}...")
        
        # Check if it looks like a prompt regurgitation
        if "Synthesize a comprehensive" in new_analysis.executive_summary:
            print("FAILURE: Summary still looks like prompt regurgitation!")
        else:
            print("SUCCESS: Summary looks generated.")
            
        # Save
        storage.save_analysis(session_id, new_analysis)
        print("Saved new analysis.")
        
    except Exception as e:
        print(f"Regeneration failed: {e}")

if __name__ == "__main__":
    session_id = "fb3c4459-7243-4050-bfe2-9d86683a82ca"
    regenerate(session_id)
