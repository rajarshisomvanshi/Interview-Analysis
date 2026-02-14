import requests
import time
import sys
import os
import json

# Configuration
API_URL = "http://localhost:8000"
VIDEO_PATH = "frontend_video.mp4"
INTERVIEWEE_NAME = "Test Candidate"
USER_ID = "test_user_flow"

def run_full_flow():
    print(f"Starting full flow test with video: {VIDEO_PATH}")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        # Try to find any mp4
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith(".mp4"):
                    print(f"Found alternative: {os.path.join(root, file)}")
                    return
        return

    # Create temp copy to avoid file lock issues
    import shutil
    import uuid
    temp_video_path = f"temp_video_{uuid.uuid4()}.mp4"
    try:
        shutil.copy2(VIDEO_PATH, temp_video_path)
        print(f"Created temp video: {temp_video_path}")
    except Exception as e:
        print(f"Failed to copy video: {e}")
        return

    # 1. Upload and Start Analysis
    print("\n1. Uploading video and starting analysis...")
    try:
        with open(temp_video_path, 'rb') as f:
            files = {'file': (os.path.basename(VIDEO_PATH), f, 'video/mp4')}
            data = {
                'interviewee_name': INTERVIEWEE_NAME,
                'user_id': USER_ID
            }
            response = requests.post(f"{API_URL}/analyze-quick", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            session_id = result['session_id']
            print(f"   Session Created: {session_id}")
            print(f"   Dashboard URL: {result['dashboard_url']}")
    except Exception as e:
        print(f"   Failed to start analysis: {e}")
        if 'response' in locals() and response is not None:
             print(f"   Response: {response.text}")
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return
        
    # Cleanup temp file early
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    # 2. Poll Status
    print("\n2. Polling status...")
    start_time = time.time()
    while True:
        try:
            status_res = requests.get(f"{API_URL}/sessions/{session_id}/status")
            status_res.raise_for_status()
            status_data = status_res.json()
            
            status = status_data['status']
            progress = status_data.get('progress', 0)
            step = status_data.get('current_step', 'Unknown')
            
            # Clear line and print status
            sys.stdout.write(f"\r   Status: {status.upper()} | Progress: {progress*100:.1f}% | Step: {step}")
            sys.stdout.flush()
            
            if status == 'completed':
                print(f"\n\n   Analysis COMPLETED in {time.time() - start_time:.1f}s")
                break
            elif status == 'failed':
                print(f"\n\n   Analysis FAILED: {status_data.get('error_message')}")
                return
            
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n   Test interrupted by user.")
            return
        except Exception as e:
            print(f"\n   Error polling status: {e}")
            time.sleep(2)

    # 3. Get Results
    print("\n3. Fetching Results...")
    try:
        results_res = requests.get(f"{API_URL}/sessions/{session_id}/results")
        results_res.raise_for_status()
        results = results_res.json()
        
        print("\n" + "="*50)
        print("EXECUTIVE SUMMARY")
        print("="*50)
        print(results.get('executive_summary', 'N/A'))
        
        print("\n" + "="*50)
        print("SCORES")
        print("="*50)
        print(f"Integrity: {results.get('integrity_score')}")
        print(f"Confidence: {results.get('confidence_score')}")
        print(f"Risk: {results.get('risk_score')}")
        
        # Check slices for detailed scores if they exist
        slices = results.get('slices', [])
        if slices:
            last_slice = slices[-1] # Take last slice as representative or average them
            print("\nDetailed UPSC Scores (from last slice):")
            print(f"Mental Alertness: {last_slice.get('mental_alertness', 'N/A')}")
            print(f"Critical Assimilation: {last_slice.get('critical_assimilation', 'N/A')}")
            print(f"Clear Exposition: {last_slice.get('clear_exposition', 'N/A')}")
            print(f"Balance of Judgment: {last_slice.get('balance_judgment', 'N/A')}")
            print(f"Depth of Interest: {last_slice.get('interest_depth', 'N/A')}")
            print(f"Social Cohesion: {last_slice.get('social_cohesion', 'N/A')}")
            print(f"Intellectual Integrity: {last_slice.get('intellectual_integrity', 'N/A')}")
            print(f"State Awareness: {last_slice.get('state_awareness', 'N/A')}")
            
    except Exception as e:
        print(f"   Failed to fetch results: {e}")

if __name__ == "__main__":
    run_full_flow()
