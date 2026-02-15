# Team Setup Instructions (Branch: interview-new)

## Prerequisites
1. **Python 3.10+** installed.
2. **FFmpeg** installed and accessible from CLI.
3. **Ollama** installed and running (`ollama serve`).

## Steps to Run the App (Backend)

1. **Clone & Switch Branch:**
   ```bash
   git clone https://github.com/dextoraai-dev/dextora_micro_services.git
   cd dextora_micro_services
   git checkout interview-new
   ```

2. **Setup Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Large Models:**
   Run the helper script to fetch necessary model files not stored in git:
   ```bash
   python download_models.py
   ```
   *Note: Other models (like `faster-whisper`, `pyannote.audio`) will download automatically on first run.*

5. **Start Server:**
   ```bash
   # Make sure `.env` is present (it is part of this branch).
   python main.py server
   ```

## Steps to Run Frontend

1. Navigate to `frontend/`:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Important Notes on Large Files
- **Models (`models/`, `*.pt`, `Placeholders`):** These are intentionally ignored `git` to keep the repo size manageable. Use `download_models.py` or check `README.md` for manual links if automation fails.
- **Session Data (`data/`):** Local session data is ignored. You will start with a fresh database.

## Environment Variables
The `.env` file is included in this branch for convenience. Modify `LLM_MODEL` or other settings locally if needed, but avoid committing local secrets.
