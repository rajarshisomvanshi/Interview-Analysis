import requests
import time
import sys

SESSION_ID = "eb9f87b9-6397-40a6-a149-fa8cb6d3b29f"
URL = f"http://localhost:8000/sessions/{SESSION_ID}/analyze"

def trigger():
    print(f"Triggering analysis for session {SESSION_ID}...")
    try:
        resp = requests.post(URL, json={})
        if resp.status_code == 200:
            print("Success! Analysis triggered.")
            print(resp.json())
        else:
            print(f"Failed: {resp.status_code}")
            print(resp.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    trigger()
