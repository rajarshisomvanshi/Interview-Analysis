# Deployment Guide

This application is designed to be easily deployed to production by separating the frontend and backend configurations.

## 1. Backend Deployment (Python/FastAPI)

The backend is a FastAPI application that requires Python 3.9+.

### Environment Variables
Set the following environment variables on your backend server (or in a `.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Server host binding | `0.0.0.0` |
| `API_PORT` | Server port | `8000` |
| `ENABLE_CORS` | Enable/Disable CORS | `True` |
| `LLM_PROVIDER` | Set to `ollama` for local Qwen | `openai` |
| `LLM_MODEL` | Model name (e.g. `qwen2.5:1.5b`) | `gpt-4` |
| `OLLAMA_BASE_URL` | URL of Ollama server | `http://localhost:11434` |

### Deployment Steps (Example: Render/Railway)
1. **Build Command**: `pip install -r requirements.txt`
2. **Start Command**: `python main.py server`
3. Ensure the server has write access to the `data/` directory for storing analysis results.

---

## 2. Frontend Deployment (React/Vite)

The frontend is a static React application that communicates with the backend via API.

### Environment Variables (Build Time)
**Crucial**: You must set these variables *during the build process* so they are baked into the static files.

| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_API_URL` | Full URL of your deployed backend. **Must not end with slash**. | `https://api.your-app.com` |

### Deployment Steps (Example: Vercel/Netlify)
1. **Build Command**: `npm run build`
2. **Output Directory**: `dist`
3. **Environment Variable**: Set `VITE_API_URL` to your live backend URL (e.g., `https://my-backend-service.onrender.com`).

### Local Testing vs Production
- **Local**: `VITE_API_URL` defaults to `/api`, which uses the Vite proxy to forward requests to `http://localhost:8000`.
- **Production**: When `VITE_API_URL` is set, the app will make direct CORS requests to that URL.

## 3. Verify Deployment
1. Open your deployed frontend URL.
2. Upload a video.
3. Check the Network tab to ensure requests are going to your `VITE_API_URL` (e.g., `https://api.your-app.com/analyze-quick`) and not `localhost`.
