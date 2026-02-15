import requests
import time
import sys

SESSION_ID = "3b67bac8-480b-4f05-9e16-f02ec5a0cd7b"
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
