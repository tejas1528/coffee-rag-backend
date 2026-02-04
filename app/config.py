"""
Configuration for FastAPI Coffee RAG Backend
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings"""
    
    # App info
    APP_NAME: str = "Coffee RAG API"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / 'data'
    MODELS_DIR: Path = DATA_DIR / 'models'
    CHROMA_DIR: Path = BASE_DIR / 'chroma_db'
    
    # Data files
    DATASET_PATH: Path = DATA_DIR / 'cotter_dataset.csv'
    RF_MODEL_PATH: Path = MODELS_DIR / 'random_forest_optimized.pkl'
    SCALER_PATH: Path = MODELS_DIR / 'scaler_optimized.pkl'
    METADATA_PATH: Path = MODELS_DIR / 'model_metadata.json'
    
    # ChromaDB
    COLLECTION_NAME: str = 'coffee_ratings'
    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080"
    ]
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    RELOAD: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()