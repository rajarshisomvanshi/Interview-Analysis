"""
Face Detection Diagnostic Script
Tests multiple face detection backends to identify what works and what doesn't.
"""

import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None

import cv2
import numpy as np
import time
from pathlib import Path

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def banner(text):
    print(f"\n{'='*60}")
    print(f"  {CYAN}{text}{RESET}")
    print(f"{'='*60}")

def ok(text):
    print(f"  {GREEN}[OK] {text}{RESET}")

def fail(text):
    print(f"  {RED}[FAIL] {text}{RESET}")

def warn(text):
    print(f"  {YELLOW}[WARN] {text}{RESET}")

def info(text):
    print(f"  {text}")

# ── Find a test video ────────────────────────────────────────
def find_test_video():
    sessions_dir = Path("data/sessions")
    if not sessions_dir.exists():
        return None
    for session in sessions_dir.iterdir():
        for name in ["source_video.mp4", "phone_video.mp4"]:
            vp = session / name
            if vp.exists() and vp.stat().st_size > 100_000:  # >100KB
                return str(vp)
    return None

# ── Extract sample frames ────────────────────────────────────
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        fail(f"Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info(f"Video: {total} frames, {fps:.1f} FPS, {w}x{h}")

    step = max(1, total // num_frames)
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frames.append((i * step, frame))
    cap.release()
    info(f"Extracted {len(frames)} sample frames")
    return frames


# ── Test 1: OpenCV Haar Cascade ──────────────────────────────
def test_opencv_haar(frames):
    banner("Test 1: OpenCV Haar Cascade (built-in)")
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            fail("Haar cascade failed to load")
            return

        ok("Haar cascade loaded")
        total_faces = 0
        t0 = time.time()
        for fnum, frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            n = len(faces)
            total_faces += n
            info(f"  Frame {fnum}: {n} face(s) detected")
        elapsed = time.time() - t0
        ok(f"Total: {total_faces} detections across {len(frames)} frames ({elapsed:.2f}s)")
    except Exception as e:
        fail(f"Haar cascade error: {e}")


# ── Test 2: OpenCV DNN (caffemodel) ──────────────────────────
def test_opencv_dnn(frames):
    banner("Test 2: OpenCV DNN Face Detector (Caffe)")
    prototxt = Path("models/deploy.prototxt")
    caffemodel = Path("models/res10_300x300_ssd_iter_140000.caffemodel")
    
    if not prototxt.exists() or not caffemodel.exists():
        warn("DNN model files not found. Skipping.")
        info(f"  Expected: {prototxt}, {caffemodel}")
        return

    try:
        net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        ok("DNN model loaded")
        total_faces = 0
        t0 = time.time()
        for fnum, frame in frames:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            n = 0
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > 0.5:
                    n += 1
            total_faces += n
            info(f"  Frame {fnum}: {n} face(s) (conf>0.5)")
        elapsed = time.time() - t0
        ok(f"Total: {total_faces} detections ({elapsed:.2f}s)")
    except Exception as e:
        fail(f"DNN error: {e}")


# ── Test 3: MediaPipe Face Detection ─────────────────────────
def test_mediapipe(frames):
    banner("Test 3: MediaPipe Face Detection")
    try:
        import mediapipe as mp
        ok("mediapipe imported")
    except ImportError:
        fail("mediapipe NOT installed")
        return

    try:
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        ok("Face detector initialized (model_selection=1 = full range)")

        total_faces = 0
        t0 = time.time()
        for fnum, frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            n = len(results.detections) if results.detections else 0
            total_faces += n
            if n > 0:
                for det in results.detections:
                    score = det.score[0] if det.score else 0
                    info(f"  Frame {fnum}: face conf={score:.2f}")
            else:
                info(f"  Frame {fnum}: 0 faces")
        elapsed = time.time() - t0
        ok(f"Total: {total_faces} detections ({elapsed:.2f}s)")
        detector.close()
    except Exception as e:
        fail(f"MediaPipe error: {e}")


# ── Test 4: face_recognition library ─────────────────────────
def test_face_recognition(frames):
    banner("Test 4: face_recognition library (dlib-based)")
    try:
        import face_recognition
        ok("face_recognition imported successfully")
    except ImportError as e:
        fail(f"face_recognition NOT installed: {e}")
        warn("This is the library used by clustering.py — this is likely your issue!")
        info("")
        info("  To install on Windows:")
        info("    pip install cmake")
        info("    pip install dlib")
        info("    pip install face_recognition")
        info("")
        info("  OR use conda:")
        info("    conda install -c conda-forge dlib")
        info("    pip install face_recognition")
        return

    try:
        total_faces = 0
        unique_encodings = []
        t0 = time.time()
        for fnum, frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            locations = face_recognition.face_locations(small, model="hog")
            n = len(locations)
            total_faces += n
            
            if locations:
                encodings = face_recognition.face_encodings(small, locations)
                for enc in encodings:
                    # Check if this is a new person
                    is_new = True
                    for existing in unique_encodings:
                        dist = np.linalg.norm(enc - existing)
                        if dist < 0.5:
                            is_new = False
                            break
                    if is_new:
                        unique_encodings.append(enc)
                        
                info(f"  Frame {fnum}: {n} face(s), {len(encodings)} encoding(s)")
            else:
                info(f"  Frame {fnum}: 0 faces")
                
        elapsed = time.time() - t0
        ok(f"Total: {total_faces} detections ({elapsed:.2f}s)")
        ok(f"Unique people estimated: {len(unique_encodings)}")
    except Exception as e:
        fail(f"face_recognition error: {e}")


# ── Test 5: YOLO (ultralytics) ────────────────────────────────
def test_yolo(frames):
    banner("Test 5: YOLO Person Detection (ultralytics)")
    try:
        from ultralytics import YOLO
        ok("ultralytics imported")
    except ImportError:
        warn("ultralytics not installed, skipping")
        return

    try:
        model = YOLO("yolov8n.pt")
        ok("YOLOv8n model loaded")
        total_persons = 0
        t0 = time.time()
        for fnum, frame in frames:
            results = model(frame, verbose=False, classes=[0])  # class 0 = person
            n = len(results[0].boxes)
            total_persons += n
            info(f"  Frame {fnum}: {n} person(s)")
        elapsed = time.time() - t0
        ok(f"Total: {total_persons} person detections ({elapsed:.2f}s)")
    except Exception as e:
        fail(f"YOLO error: {e}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    banner("FACE DETECTION DIAGNOSTIC")
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else find_test_video()
    if not video_path:
        fail("No test video found. Pass path as argument: python test_face_detection.py <video.mp4>")
        sys.exit(1)
    
    ok(f"Test video: {video_path}")
    frames = extract_frames(video_path, num_frames=8)
    
    if not frames:
        fail("Could not extract frames")
        sys.exit(1)
    
    # Save a sample frame for visual inspection
    sample_frame = frames[len(frames)//2][1]
    cv2.imwrite("test_face_sample_frame.jpg", sample_frame)
    info(f"Saved sample frame to test_face_sample_frame.jpg")
    
    test_opencv_haar(frames)
    test_mediapipe(frames)
    test_face_recognition(frames)
    test_opencv_dnn(frames)
    test_yolo(frames)
    
    banner("SUMMARY & RECOMMENDATIONS")
    print()
    info("If face_recognition fails: Your clustering.py falls back to")
    info("a single 'person_0' identity — no multi-person detection.")
    print()
    info("QUICKEST FIX: Use MediaPipe face detection (already installed)")
    info("as a fallback when face_recognition is unavailable.")
    info("MediaPipe can detect faces but NOT generate embeddings for")
    info("identity matching. For clustering, you need either:")
    info("  1. Install face_recognition (pip install face_recognition)")
    info("  2. Use FaceNet/ArcFace via deepface library")
    info("  3. Use MediaPipe for detection + a custom embedding model")
    print()
