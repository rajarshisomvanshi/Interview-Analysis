# Paper Mimic API

Standalone API for processing exam PDFs and generating mimic questions.

## Setup


2.  **Environment Setup**
    If you pulled this repo, you need to set up the virtual environment:
    ```bash
    # Create virtual environment
    python -m venv .venv

    # Activate virtual environment (Windows)
    .venv\Scripts\activate

    # Activate virtual environment (Mac/Linux)
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration**
    Ensure `.env` exists with valid `GEMINI_API_KEY`.

3.  **Run Server**
    ```bash
    uvicorn src.main:app --host 0.0.0.0 --port 2000
    ```

## Usage

**Endpoint:** `POST /api/process`

**Body:** `multipart/form-data` with `file` field (PDF).

**Response:**
```json
{
  "status": 1,
  "data": {
    "parsed_text_length": 1234,
    "questions_found": 5,
    "generated_results": [...]
  }
}
```

## Docker

```bash
docker-compose up --build
```
