
import sys
import os
import uuid
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from core.storage import StorageManager
from core.schemas import SessionData, SessionMetadata, SessionAnalysis, QuestionAnalysis

def reproduce_persistence_issue():
    print("Starting persistence reproduction test...")
    storage = StorageManager()
    session_id = str(uuid.uuid4())
    print(f"Test Session ID: {session_id}")

    # 1. Create and Save Session (Processing)
    metadata = SessionMetadata(
        session_id=session_id,
        created_at=datetime.utcnow(),
        interviewee_name="Test Subject",
        status="processing",
        audio_path="dummy_audio.wav"
    )
    session_data = SessionData(metadata=metadata)
    storage.save_session_data(session_data)
    print("1. Saved initial session data (processing).")

    # 2. Update with Analysis (Completed)
    analysis = SessionAnalysis(
        session_id=session_id,
        analyzed_at=datetime.utcnow(),
        executive_summary="This is a persistent summary.",
        integrity_score=85.0,
        confidence_score=75.0,
        risk_score=15.0,
        question_analyses=[
            QuestionAnalysis(
                qa_index=0,
                communication_clarity="Clear",
                confidence_indicators="High",
                stress_indicators="None",
                summary="Good answer",
                analysis_confidence="high"
            )
        ]
    )
    
    session_data.metadata.status = "completed"
    session_data.analysis = analysis # Link it
    
    # Save both
    storage.save_session_data(session_data)
    storage.save_analysis(session_id, analysis)
    print("2. Saved completed session data and analysis.")

    # 3. Simulate "Re-open" (Load from disk)
    print("-" * 20)
    print("3. reloading session data from disk...")
    
    loaded_session = storage.load_session_data(session_id)
    
    if loaded_session:
        print(f"Loaded Status: {loaded_session.metadata.status}")
        if loaded_session.analysis:
            print("SUCCESS: SessionData.analysis is PRESENT.")
            print(f"Summary: {loaded_session.analysis.executive_summary}")
            print(f"Score: {loaded_session.analysis.integrity_score}")
        else:
            print("FAILURE: SessionData.analysis is MISSING.")
            
            # Try separate load
            loaded_analysis = storage.load_analysis(session_id)
            if loaded_analysis:
                print("However, load_analysis() succeeded separately.")
            else:
                print("FATAL: load_analysis() also failed.")
    else:
        print("FATAL: Could not load session data.")

    # 4. Cleanup
    # shutil.rmtree(storage.get_session_dir(session_id)) 
    print("Test complete.")

if __name__ == "__main__":
    reproduce_persistence_issue()
