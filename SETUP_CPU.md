# CPU-Only Setup Guide

Quick setup guide for running the Interview Intelligence System on CPU with Phi3 Mini.

## Prerequisites

- Python 3.9+
- Ollama installed
- At least 8GB RAM
- FFmpeg (for video/audio processing)

---

## Step 1: Install Ollama

### Windows
```powershell
# Download and install from ollama.com
# Or use winget:
winget install Ollama.Ollama
```

### Verify installation
```bash
ollama --version
```

---

## Step 2: Pull Phi3 Mini Model

```bash
ollama pull phi3:mini
```

**Model details:**
- Size: ~2.3GB
- Parameters: 3.8B
- Speed: Very fast on CPU (~15-20 seconds per question)
- Quality: Good for behavioral analysis

---

## Step 3: Install Python Dependencies

```bash
cd c:\Users\amart\Documents\projects\YOLO-cctv-interview

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Some packages may take time to install. Key packages:
- `ultralytics` (YOLOv8)
- `faster-whisper` (Speech-to-text)
- `pyannote.audio` (Speaker diarization)
- `mediapipe` (Face mesh)
- `fastapi` (API server)

---

## Step 4: Verify Configuration

The `.env` file is already configured for CPU with Phi3 Mini. Verify settings:

```bash
type .env
```

Key settings:
- `LLM_MODEL=phi3:mini`
- `USE_GPU=false`
- `VIDEO_FPS=15`
- `WHISPER_MODEL=tiny`
- `YOLO_PHONE_MODEL=yolov8n.pt`

---

## Step 5: Test Ollama Connection

```bash
# Test Phi3 Mini
ollama run phi3:mini "Analyze this behavioral pattern: increased blink rate during technical questions. Use cautious language."
```

Expected response: Should provide analysis using phrases like "may indicate", "suggests", etc.

---

## Step 6: Run the Server

```bash
python main.py server
```

Server will start on `http://localhost:8000`

Check health:
```bash
curl http://localhost:8000/health
```

---

## Step 7: Test with Sample Data

### Create a test session
```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"interviewee_name": "Test User"}'
```

Response will include `session_id` and upload paths.

### Upload files (replace {session_id})
```bash
# Upload phone video
curl -X POST http://localhost:8000/sessions/{session_id}/upload/phone-video \
  -F "file=@path/to/phone_video.mp4"

# Upload CCTV video
curl -X POST http://localhost:8000/sessions/{session_id}/upload/cctv-video \
  -F "file=@path/to/cctv_video.mp4"

# Upload audio
curl -X POST http://localhost:8000/sessions/{session_id}/upload/audio \
  -F "file=@path/to/audio.wav"
```

### Trigger analysis
```bash
curl -X POST http://localhost:8000/sessions/{session_id}/analyze
```

### Check status
```bash
curl http://localhost:8000/sessions/{session_id}/status
```

### Get results (when complete)
```bash
curl http://localhost:8000/sessions/{session_id}/results
```

---

## Expected Performance (CPU)

For a **10-minute interview** with **10 questions**:

| Component | Time | Notes |
|-----------|------|-------|
| Vision (YOLO) | ~4-6 min | YOLOv8n at 15 FPS |
| Audio (Whisper) | ~1-2 min | Whisper tiny model |
| Diarization | ~1-2 min | pyannote.audio |
| LLM Analysis | ~2-3 min | Phi3 mini (~15s/question) |
| **Total** | **~8-13 min** | Acceptable for batch processing |

---

## Troubleshooting

### Ollama not found
```bash
# Check if Ollama is running
ollama list

# Start Ollama service (Windows)
# It should auto-start, but if not:
# Restart Ollama from Start Menu
```

### Slow processing
- Reduce `VIDEO_FPS` to 10 in `.env`
- Use even smaller Whisper: `WHISPER_MODEL=tiny.en` (English-only)
- Reduce `NUM_WORKERS` if CPU is overloaded

### Memory issues
- Close other applications
- Reduce `BATCH_SIZE` to 1 (already set)
- Use `WHISPER_MODEL=tiny` (already set)

### Pyannote.audio errors
If you get authentication errors:
```bash
# Get HuggingFace token from huggingface.co/settings/tokens
# Set environment variable:
set HF_TOKEN=your_token_here

# Or add to .env:
# HF_TOKEN=your_token_here
```

---

## Performance Optimization Tips

1. **Lower video FPS**: Set `VIDEO_FPS=10` for 30% faster processing
2. **Skip video storage**: Set `ENABLE_VIDEO_STORAGE=false` to save disk I/O
3. **Increase workers**: Set `NUM_WORKERS` to your CPU core count (check with `wmic cpu get NumberOfCores`)
4. **Use English-only Whisper**: `WHISPER_MODEL=tiny.en` if interviews are in English

---

## Next Steps

1. Prepare sample interview data (phone video, CCTV video, audio)
2. Test the complete pipeline
3. Review analysis results
4. Adjust settings based on performance

For full documentation, see [README.md](README.md)
