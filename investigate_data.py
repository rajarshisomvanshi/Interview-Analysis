import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.getcwd())

from core.storage import StorageManager
from core.schemas import SessionData, SessionAnalysis

def investigate(session_id):
    print(f"Investigating session {session_id}...")
    storage = StorageManager()
    
    # 1. Load Session Data
    try:
        session_data = storage.load_session_data(session_id)
        if session_data:
            print("SessionData loaded.")
            if session_data.analysis:
                print("SUCCESS: SessionData.analysis is PRESENT.")
                print(f"Summary length: {len(session_data.analysis.executive_summary)}")
            else:
                print("FAILURE: SessionData.analysis is MISSING.")
        else:
            print("SessionData not found.")
    except Exception as e:
        print(f"Error loading SessionData: {e}")

    # 2. Load Analysis Results
    try:
        analysis = storage.load_analysis(session_id)
        if analysis:
            print("AnalysisResults loaded.")
            print(f"Summary from file: {analysis.executive_summary[:50]}...")
        else:
            print("AnalysisResults not found.")
    except Exception as e:
        print(f"Error loading AnalysisResults: {e}")
        
    # 3. Try to merge
    if session_data and not session_data.analysis and analysis:
        print("Attempting to merge analysis into session_data...")
        session_data.analysis = analysis
        try:
            # Check serialization
            dump = session_data.model_dump(mode='json')
            if dump.get('analysis'):
                print("Merge successful in model_dump.")
                # storage.save_session_data(session_data) # Uncomment to fix
                print("Save check passed (simulated).")
            else:
                print("Merge FAILED in model_dump.")
        except Exception as e:
            print(f"Merge failed: {e}")

if __name__ == "__main__":
    session_id = "fb3c4459-7243-4050-bfe2-9d86683a82ca"
    investigate(session_id)
