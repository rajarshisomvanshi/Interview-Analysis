import requests
import time
import sys
import argparse

# Default ID if none provided
DEFAULT_SESSION_ID = "3b67bac8-480b-4f05-9e16-f02ec5a0cd7b"

def trigger(session_id):
    url = f"http://localhost:8000/sessions/{session_id}/analyze"
    print(f"Triggering analysis for session {session_id}...")
    try:
        resp = requests.post(url, json={})
        if resp.status_code == 200:
            print("Success! Analysis triggered.")
            print(resp.json())
        else:
            print(f"Failed: {resp.status_code}")
            print(resp.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrigger analysis for a session')
    parser.add_argument('--session_id', type=str, default=DEFAULT_SESSION_ID, help='Session ID to retrigger')
    
    args = parser.parse_args()
    trigger(args.session_id)
