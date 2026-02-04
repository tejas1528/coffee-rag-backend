"""
FastAPI Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.rag_engine import CoffeeRAGEngine
from app.ml_predictor import MLPredictor
# DON'T import routers here - move them down

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
rag_engine = None
ml_predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global rag_engine, ml_predictor
    
    # Startup
    logger.info("Starting Coffee RAG API...")
    
    # Initialize ML Predictor
    ml_predictor = MLPredictor(settings)
    ml_predictor.load_models()
    
    # Initialize RAG Engine
    rag_engine = CoffeeRAGEngine(settings)
    await rag_engine.initialize()
    
    logger.info("✓ All systems initialized!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection functions (BEFORE importing routers)
def get_rag_engine():
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return rag_engine

def get_ml_predictor():
    if ml_predictor is None:
        raise HTTPException(status_code=503, detail="ML predictor not loaded")
    return ml_predictor

# NOW import routers (after dependency functions are defined)
from app.routers import recipe, prediction, analysis

# Include routers
app.include_router(recipe.router, prefix=settings.API_V1_PREFIX)
app.include_router(prediction.router, prefix=settings.API_V1_PREFIX)
app.include_router(analysis.router, prefix=settings.API_V1_PREFIX)

# Root endpoint
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "rag_engine": "initialized" if rag_engine else "not initialized",
        "ml_predictor": "loaded" if ml_predictor else "not loaded"
    }