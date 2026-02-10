
import urllib.request
import os

url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
output_path = "models/face_landmarker.task"

print(f"Downloading {url} to {output_path}...")
try:
    urllib.request.urlretrieve(url, output_path)
    print("Download complete.")
except Exception as e:
    print(f"Download failed: {e}")
