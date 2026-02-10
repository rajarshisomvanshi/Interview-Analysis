# Developer Handoff Guide: UPSC Interview Intelligence System

This document provides technical instructions for extending and deploying the "UPSC Forensic" intelligence layer.

## Architectural Overview
The system is built on a modular "Forensic Layer" that separates raw visual/vocal telemetry from semantic behavioral insights.

### 1. Forensic API Framework
New granular endpoints are available in `api/routes.py`:
- `GET /sessions/{id}/signals`: Returns time-stamped vectors (face landmarks, emotions, vocal telemetry).
- `GET /sessions/{id}/events`: Returns semantic forensic events (e.g., "Postural Shift", "Hesitation Pause").
- `GET /sessions/{id}/scores/forensic`: Returns the weighted Board Impression and Risk scores.

### 2. Cognitive Memory (Vestige)
We use the **Vestige MCP Server** (https://github.com/samvallad33/vestige) for cross-session behavioral memory.
- **Client**: `core/memory.py` handles ingestion and retrieval.
- **Spreading Activation**: The system can retrieve past stress patterns even if the query is non-exact.
- **Setup**:
    1. Download the `vestige-mcp` binary for your OS from the Vestige GitHub releases.
    2. Add to your path or configure the `mcp_command` in `MemoryClient`.
    3. The `InterviewAnalyzer` will automatically use it for forensic RAG context.

## Deployment Instructions

### Frontend Environment
The frontend is now environment-agnostic. To deploy to a production domain:
1.  Copy `frontend/.env.example` to `frontend/.env`.
2.  Update `VITE_API_URL` to your production backend URL:
    ```
    VITE_API_URL=https://api.your-interview-system.com
    ```
3.  Build the frontend: `npm run build`.

### Backend Weights
Forensic scores are currently calculated using a weighted average in `routes.py`:
- Visual: 40%
- Vocal: 30%
- Linguistic: 30%
These can be adjusted via the `/recalculate` endpoint (placeholder implemented).

## Stability & Safety
The new "Forensic Layer" consists of additive endpoints. Existing `/status` and `/results` endpoints remain untouched to ensure backward compatibility.

---
**Prepared by Antigravity AI**
*Forensic Behavior Analysis Framework v1.0*
