
import requests
import sys

URL = "http://127.0.0.1:8000/analyze-quick"
VIDEO_PATH = "frontend_video.mp4"

def test_upload():
    print(f"Testing upload to {URL} with {VIDEO_PATH}...")
    try:
        with open(VIDEO_PATH, 'rb') as f:
            files = {'file': (VIDEO_PATH, f, 'video/mp4')}
            data = {
                'interviewee_name': 'Test Candidate',
                'user_id': 'test_user_123'
            }
            response = requests.post(URL, files=files, data=data, timeout=30)
            
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Upload successful!")
        else:
            print("Upload failed!")
            
    except Exception as e:
        print(f"Exception during upload: {e}")

if __name__ == "__main__":
    test_upload()
