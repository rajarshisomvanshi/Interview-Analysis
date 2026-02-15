import time
import requests
import sys
import os

API_BASE = "http://localhost:8000"

def get_latest_session():
    sessions_dir = "data/sessions"
    if not os.path.exists(sessions_dir):
        return None
    sessions = [os.path.join(sessions_dir, d) for d in os.listdir(sessions_dir) if os.path.isdir(os.path.join(sessions_dir, d))]
    if not sessions:
        return None
    latest_session = max(sessions, key=os.path.getmtime)
    return os.path.basename(latest_session)

def monitor_session(session_id):
    print(f"Monitoring session: {session_id}")
    url = f"{API_BASE}/sessions/{session_id}/status"
    
    last_status = ""
    last_progress = -1

    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                progress = data.get("progress", 0.0)
                if progress is None: progress = 0.0
                
                current_step = data.get("current_step", "")
                
                # Format progress bar
                bar_len = 30
                filled_len = int(bar_len * progress)
                bar = '=' * filled_len + '-' * (bar_len - filled_len)
                
                sys.stdout.write(f"\r[{bar}] {int(progress * 100)}% - Status: {status} - Step: {current_step}   ")
                sys.stdout.flush()
                
                if status in ["completed", "failed"]:
                    print(f"\nAnalysis {status}!")
                    break
            else:
                print(f"\rError fetching status: {response.status_code}", end="")
        except Exception as e:
            print(f"\rConnection error: {e}", end="")
            
        time.sleep(2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
    else:
        # Focusing on the new correct session
        session_id = "3b67bac8-480b-4f05-9e16-f02ec5a0cd7b"
    
    if session_id:
        monitor_session(session_id)
    else:
        print("No active session found.")
