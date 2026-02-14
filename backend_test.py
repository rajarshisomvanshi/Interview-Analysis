import requests
import time
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"
# Use the same video path as before
VIDEO_PATH = "frontend_video.mp4"

def test_analyze_quick():
    print(f"Triggering quick analysis for: {VIDEO_PATH}")
    files = {'file': open(VIDEO_PATH, 'rb')}
    response = requests.post(f"{BASE_URL}/analyze-quick", files=files)
    
    if response.status_code == 200:
        session_id = response.json().get("session_id")
        print(f"Analysis started. Session ID: {session_id}")
        return session_id
    else:
        print(f"Failed to start analysis: {response.text}")
        return None

def monitor_status(session_id):
    while True:
        response = requests.get(f"{BASE_URL}/sessions/{session_id}/status")
        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            progress = data.get("progress", 0)
            step = data.get("current_step", "Unknown")
            
            print(f"Status: {status} | Progress: {progress:.2%} | Step: {step}")
            
            if status == "completed":
                print("Analysis COMPLETED!")
                return True
            elif status == "failed":
                print(f"Analysis FAILED: {data.get('error_message')}")
                return False
        else:
            print(f"Failed to get status: {response.text}")
            return False
        
        time.sleep(10)

if __name__ == "__main__":
    session_id = test_analyze_quick()
    if session_id:
        monitor_status(session_id)
