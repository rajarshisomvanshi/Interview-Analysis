import json
import os
from pathlib import Path

def list_sessions():
    sessions_dir = Path("data/sessions")
    if not sessions_dir.exists():
        print("Sessions directory not found.")
        return
        
    for s in os.listdir(sessions_dir):
        path = sessions_dir / s / "session_data.json"
        if path.exists():
            with open(path, 'r') as f:
                try:
                    data = json.load(f)
                    metadata = data.get("metadata", {})
                    ts = len(data.get("transcript_segments", []))
                    qa = len(data.get("question_answer_pairs", []))
                    status = metadata.get("status", "unknown")
                    print(f"Session {s}: Status={status}, Transcript={ts}, Q&A={qa}")
                except Exception as e:
                    print(f"Error reading session {s}: {e}")

if __name__ == "__main__":
    list_sessions()
