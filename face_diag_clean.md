`

============================================================
  FACE DETECTION DIAGNOSTIC
============================================================
  [OK] Test video: data\sessions\15308a04-81d4-4bd7-a8f1-1706e806c26b\phone_video.mp4
  Video: 52829 frames, 25.0 FPS, 640x360
  Extracted 8 sample frames
  Saved sample frame to test_face_sample_frame.jpg

============================================================
  Test 1: OpenCV Haar Cascade (built-in)
============================================================
  [OK] Haar cascade loaded
    Frame 0: 0 face(s) detected
    Frame 6603: 1 face(s) detected
    Frame 13206: 1 face(s) detected
    Frame 19809: 1 face(s) detected
    Frame 26412: 1 face(s) detected
    Frame 33015: 2 face(s) detected
    Frame 39618: 1 face(s) detected
    Frame 46221: 0 face(s) detected
  [OK] Total: 7 detections across 8 frames (0.37s)

============================================================
  Test 3: MediaPipe Face Detection
============================================================
  [OK] mediapipe imported
  [FAIL] MediaPipe error: module 'mediapipe' has no attribute 'solutions'

============================================================
  Test 4: face_recognition library (dlib-based)
============================================================
  [FAIL] face_recognition NOT installed: No module named 'face_recognition'
  [WARN] This is the library used by clustering.py — this is likely your issue!
  
    To install on Windows:
      pip install cmake
      pip install dlib
      pip install face_recognition
  
    OR use conda:
      conda install -c conda-forge dlib
      pip install face_recognition

============================================================
  Test 2: OpenCV DNN Face Detector (Caffe)
============================================================
  [WARN] DNN model files not found. Skipping.
    Expected: models\deploy.prototxt, models\res10_300x300_ssd_iter_140000.caffemodel

============================================================
  Test 5: YOLO Person Detection (ultralytics)
============================================================
  [OK] ultralytics imported
  [OK] YOLOv8n model loaded
    Frame 0: 0 person(s)
    Frame 6603: 1 person(s)
    Frame 13206: 1 person(s)
    Frame 19809: 1 person(s)
    Frame 26412: 1 person(s)
    Frame 33015: 1 person(s)
    Frame 39618: 1 person(s)
    Frame 46221: 6 person(s)
  [OK] Total: 12 person detections (0.85s)

============================================================
  SUMMARY & RECOMMENDATIONS
============================================================

  If face_recognition fails: Your clustering.py falls back to
  a single 'person_0' identity — no multi-person detection.

  QUICKEST FIX: Use MediaPipe face detection (already installed)
  as a fallback when face_recognition is unavailable.
  MediaPipe can detect faces but NOT generate embeddings for
  identity matching. For clustering, you need either:
    1. Install face_recognition (pip install face_recognition)
    2. Use FaceNet/ArcFace via deepface library
    3. Use MediaPipe for detection + a custom embedding model


---STDERR---

`
