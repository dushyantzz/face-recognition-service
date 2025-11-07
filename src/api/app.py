"""FastAPI application factory"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.api.routes import detection, recognition, identity, health
from src.database.db_manager import DatabaseManager
from src.detection.retinaface_detector import RetinaFaceDetector
from src.recognition.arcface_embedder import ArcFaceEmbedder
from src.matching.matcher import FaceMatcher

logger = logging.getLogger(__name__)

# Global instances
db_manager = None
detector = None
embedder = None
matcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global db_manager, detector, embedder, matcher
    
    logger.info("Initializing Face Recognition Service...")
    
    try:
        # Initialize database
        db_manager = DatabaseManager(settings.DATABASE_PATH)
        await db_manager.initialize()
        logger.info("Database initialized")
        
        # Initialize detector
        detector = RetinaFaceDetector(
            model_path=settings.DETECTION_MODEL_PATH,
            confidence_threshold=settings.DETECTION_THRESHOLD,
            device=settings.DEVICE
        )
        logger.info(f"Face detector initialized (device: {settings.DEVICE})")
        
        # Initialize embedder
        embedder = ArcFaceEmbedder(
            model_path=settings.RECOGNITION_MODEL_PATH,
            device=settings.DEVICE
        )
        logger.info("Face embedder initialized")
        
        # Initialize matcher
        matcher = FaceMatcher(
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
            top_k=settings.TOP_K
        )
        logger.info("Face matcher initialized")
        
        # Load existing embeddings
        embeddings = await db_manager.get_all_embeddings()
        if embeddings:
            for emb in embeddings:
                matcher.add_embedding(emb['id'], emb['embedding'], emb['name'])
            logger.info(f"Loaded {len(embeddings)} embeddings into matcher")
        
        logger.info("Face Recognition Service ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Face Recognition Service...")
    if db_manager:
        await db_manager.close()
    logger.info("Service shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title=settings.APP_NAME,
        description="Production-ready Face Recognition Service with RetinaFace detection and ArcFace embeddings",
        version=settings.VERSION,
        docs_url=f"{settings.API_PREFIX}/docs",
        redoc_url=f"{settings.API_PREFIX}/redoc",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(
        health.router,
        prefix=settings.API_PREFIX,
        tags=["health"]
    )
    app.include_router(
        detection.router,
        prefix=settings.API_PREFIX,
        tags=["detection"]
    )
    app.include_router(
        recognition.router,
        prefix=settings.API_PREFIX,
        tags=["recognition"]
    )
    app.include_router(
        identity.router,
        prefix=settings.API_PREFIX,
        tags=["identity"]
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Global exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)}
        )
    
    return app


# For direct running
if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
