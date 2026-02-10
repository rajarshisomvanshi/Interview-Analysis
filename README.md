# Interview Intelligence System

A production-grade multimodal AI system for analyzing IAS-style interviews using synchronized video and audio streams.

## System Overview

The Interview Intelligence System processes three synchronized input streams:
1. **Phone camera video**: Face-focused capture (~2m distance, zoom lens)
2. **CCTV camera video**: Full-body movement capture
3. **Microphone audio**: Dual-channel capture (interviewer + interviewee)

### Core Design Principle

**Vision and audio models extract structured signals → LLM reasons over time-aligned multimodal data**

The LLM never processes raw video/audio directly. Instead, it receives structured, timestamped behavioral and communication signals aligned on a unified timeline.

---

## Features

### Vision Pipeline
- **YOLO-based detection**: Face detection (phone) and pose estimation (CCTV)
- **Facial analysis**: Face recognition, eye contact, blink rate, Action Units (MediaPipe)
- **Body movement**: Hand fidgeting, posture shifts, leg movement tracking

### Audio Pipeline
- **Speaker diarization**: Separate interviewer and interviewee
- **Speech-to-text**: Word-level timestamps using faster-whisper
- **Question-answer segmentation**: Automatic Q&A pair extraction
- **Audio signals**: Speech rate, pitch stability, filler words, fluency metrics

### Timeline Fusion
- **Unified timeline**: Millisecond-precision alignment of all signals
- **Per-question aggregation**: Combine face, body, and audio signals for each Q&A pair

### LLM Reasoning
- **Behavioral analysis**: Per-question and session-level insights
- **Cautious language**: No emotion or deception claims, only observable patterns
- **Multi-provider support**: OpenAI, Anthropic, or Ollama

### Storage & API
- **Retention policies**: Permanent JSON storage, optional video retention (7 days default)
- **RESTful API**: FastAPI with async processing
- **Progress tracking**: Real-time analysis status updates

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (optional, for faster processing)
- FFmpeg (for video/audio processing)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd YOLO-cctv-interview
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Download models** (automatic on first run)
- YOLO models will download automatically
- Whisper models will download on first use
- For pyannote.audio, you may need a HuggingFace token

---

## Usage

### Running the API Server

```bash
python main.py server
```

The server will start on `http://localhost:8000` (configurable in `.env`).

### API Endpoints

#### 1. Create Session
```bash
POST /sessions
Content-Type: application/json

{
  "interviewee_name": "John Doe",
  "interviewee_id": "12345"
}
```

Response:
```json
{
  "session_id": "uuid",
  "upload_urls": {
    "phone_video": "/path/to/phone_video.mp4",
    "cctv_video": "/path/to/cctv_video.mp4",
    "audio": "/path/to/audio.wav"
  },
  "created_at": "2026-02-06T18:00:00Z"
}
```

#### 2. Upload Files
```bash
POST /sessions/{session_id}/upload/phone-video
Content-Type: multipart/form-data

file: <phone_video.mp4>
```

Repeat for CCTV video and audio:
- `POST /sessions/{session_id}/upload/cctv-video`
- `POST /sessions/{session_id}/upload/audio`

#### 3. Trigger Analysis
```bash
POST /sessions/{session_id}/analyze
```

Response:
```json
{
  "status": "processing",
  "session_id": "uuid"
}
```

#### 4. Check Status
```bash
GET /sessions/{session_id}/status
```

Response:
```json
{
  "session_id": "uuid",
  "status": "processing",
  "progress": 0.7,
  "current_step": "Performing LLM analysis"
}
```

#### 5. Get Results
```bash
GET /sessions/{session_id}/results
```

Response:
```json
{
  "session_id": "uuid",
  "analyzed_at": "2026-02-06T18:30:00Z",
  "question_count": 10,
  "session_duration_ms": 600000,
  "executive_summary": "Behavioral analysis summary...",
  "download_url": "/sessions/{session_id}/download/analysis"
}
```

---

## Architecture

```
interview-intelligence-system/
├── config/          # Configuration management
├── core/            # Core schemas, timeline, storage
├── vision/          # YOLO detection, face/body analysis
├── audio/           # Diarization, transcription, segmentation
├── fusion/          # Timeline alignment, signal aggregation
├── reasoning/       # LLM prompts, analysis, constraints
├── api/             # FastAPI application
├── utils/           # Video/audio utilities
├── data/            # Session data storage
└── main.py          # Entry point
```

---

## Configuration

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai/anthropic/ollama) | `openai` |
| `LLM_MODEL` | Model name | `gpt-4` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `VIDEO_FPS` | Target FPS for processing | `30` |
| `VIDEO_RETENTION_DAYS` | Days to keep raw videos | `7` |
| `USE_GPU` | Enable GPU acceleration | `true` |

---

## Edge Deployment

For edge deployment, consider:

1. **Model quantization**: Enable `ENABLE_MODEL_QUANTIZATION=true`
2. **Lightweight models**: Use `yolov8n` and `whisper-base`
3. **Reduce FPS**: Set `VIDEO_FPS=15` for faster processing
4. **Disable video storage**: Set `ENABLE_VIDEO_STORAGE=false`

---

## Behavioral Analysis Constraints

The system enforces strict language constraints:

✅ **Allowed**: Observable behavioral patterns
- "Increased blink rate during technical questions"
- "Response latency averaged 2.3 seconds"
- "Eye contact decreased in final third of interview"

❌ **Prohibited**: Emotion or deception claims
- "The candidate is nervous"
- "They are lying"
- "Feels confident"

All outputs use cautious language: "may indicate", "suggests", "observable pattern".

---

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Structure
- **Modular design**: Each component (vision, audio, fusion, reasoning) is independent
- **Type safety**: Pydantic schemas for all data structures
- **Async support**: FastAPI with background tasks for long-running analysis

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `BATCH_SIZE` in `.env`
   - Use smaller models (`yolov8n`, `whisper-tiny`)

2. **Pyannote.audio authentication**
   - Set `HF_TOKEN` environment variable with HuggingFace token
   - Accept model license on HuggingFace

3. **Slow processing**
   - Enable GPU: `USE_GPU=true`
   - Reduce video FPS: `VIDEO_FPS=15`
   - Use faster-whisper instead of standard whisper

---

## License

[Specify your license here]

---

## Citation

If you use this system in research, please cite:

```
[Add citation information]
```

---

## Contact

For questions or support, contact: [your-email@example.com]
