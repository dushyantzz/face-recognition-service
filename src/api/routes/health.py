"""Health check endpoints"""

import psutil
import platform
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

from src.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    system_info: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        timestamp=datetime.utcnow().isoformat(),
        system_info={
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "device": settings.DEVICE
        }
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    from src.api.app import db_manager, detector, embedder, matcher
    
    if not all([db_manager, detector, embedder, matcher]):
        return {"status": "not_ready"}, 503
    
    return {"status": "ready"}
