"""
Interview Intelligence System - FastAPI Application

Main FastAPI application setup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Force reload triggers
from config import settings
from api.routes import router
import sys
import importlib.util

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Interview Intelligence System",
    description="Production-grade multimodal AI system for analyzing IAS-style interviews",
    version="1.0.0"
)

# Add CORS middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API routes
app.include_router(router)

from fastapi.staticfiles import StaticFiles
# Mount data directory for static access (video playback)
app.mount("/data", StaticFiles(directory=settings.data_dir), name="data")


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Interview Intelligence System API")
    logger.info(f"Executable: {sys.executable}")
    logger.info(f"Data directory: {settings.data_dir}")
    logger.info(f"LLM provider: {settings.llm_provider}/{settings.llm_model}")
    
    # Create data directory if it doesn't exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Interview Intelligence System API")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Interview Intelligence System",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with diagnostics"""
    packages = ["torch", "ultralytics", "sklearn", "pyannote.audio", "cv2", "mediapipe", "faster_whisper"]
    package_status = {}
    for p in packages:
        package_status[p] = "installed" if importlib.util.find_spec(p) else "not_installed"
    
    return {
        "status": "healthy",
        "executable": sys.executable,
        "packages": package_status,
        "python_version": sys.version
    }
