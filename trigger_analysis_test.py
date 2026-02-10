
import requests
import time
import os
from pathlib import Path

def trigger_analysis():
    url = "http://localhost:8000/analyze-quick"
    # Point to the video file in root
    video_path = Path("videoplayback (2).mp4")
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    files = {
        'file': ('source_video.mp4', open(video_path, 'rb'), 'video/mp4')
    }
    data = {
        'interviewee_name': 'Test User',
        'user_id': 'test_user_' + str(int(time.time()))
    }
    
    try:
        print(f"Uploading video from {video_path}...")
        response = requests.post(url, files=files, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    trigger_analysis()
