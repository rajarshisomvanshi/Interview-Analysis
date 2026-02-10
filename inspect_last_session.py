
import os
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data/sessions")

def inspect_latest_session():
    if not DATA_DIR.exists():
        print("No sessions directory found.")
        return

    # Find latest session directory
    sessions = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not sessions:
        print("No sessions found.")
        return

    latest_session = sessions[0]
    print(f"Latest Session: {latest_session.name}")
    
    data_file = latest_session / "session_data.json"
    if not data_file.exists():
        print("No session_data.json found.")
        return

    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
            
        print("Session Data Overview:")
        print(f"  Status: {data.get('metadata', {}).get('status')}")
        
        analysis = data.get('analysis')
        if not analysis:
            print("  [ERROR] No 'analysis' object found in session data.")
        else:
            slices = analysis.get('slices', [])
            print(f"  Slices Count: {len(slices)}")
            if slices:
                print("  Sample Slice 1:", json.dumps(slices[0], indent=2))
            else:
                print("  [WARNING] 'slices' list is empty.")
                
    except Exception as e:
        print(f"Error reading session data: {e}")

if __name__ == "__main__":
    inspect_latest_session()
