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
from app.routers import recipe, prediction, analysis

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

# Include routers
app.include_router(recipe.router, prefix=settings.API_V1_PREFIX)
app.include_router(prediction.router, prefix=settings.API_V1_PREFIX)
app.include_router(analysis.router, prefix=settings.API_V1_PREFIX)

# Dependency injection
def get_rag_engine():
    return rag_engine

def get_ml_predictor():
    return ml_predictor

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_engine": "initialized" if rag_engine else "not initialized",
        "ml_predictor": "loaded" if ml_predictor else "not loaded"
    }