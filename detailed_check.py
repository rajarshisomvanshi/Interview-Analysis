
import sys
import traceback

def check(name):
    print(f"--- Checking {name} ---")
    try:
        mod = __import__(name)
        print(f"OK: {name} SUCCESS")
        if hasattr(mod, '__version__'):
            print(f"   Version: {mod.__version__}")
        elif hasattr(mod, 'VERSION'):
             print(f"   Version: {mod.VERSION}")
    except Exception as e:
        print(f"FAIL: {name} FAILED: {e}")
        traceback.print_exc()
    print()

packages = ["torch", "ultralytics", "sklearn", "pyannote.audio", "cv2", "mediapipe", "faster_whisper"]
for p in packages:
    check(p)
