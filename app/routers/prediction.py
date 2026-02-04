"""
Prediction router - ML predictions
"""
from fastapi import APIRouter, HTTPException, Depends
from app.schemas import PredictionRequest, PredictionResponse
from app.main import get_ml_predictor, get_rag_engine
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["ML Predictions"])

@router.post("/", response_model=PredictionResponse)
async def predict_liking(
    request: PredictionRequest,
    ml_predictor = Depends(get_ml_predictor),
    rag_engine = Depends(get_rag_engine)
):
    """Predict coffee liking score from sensory features"""
    
    try:
        # Get RF prediction
        result = ml_predictor.predict_with_explanation(request.sensory_features)
        
        # Optionally add RAG context
        rag_context = None
        combined_prediction = None
        
        if request.user_preference:
            # Get similar coffees for context
            similar = rag_engine.find_similar_coffees(request.user_preference, n_results=5)
            
            if similar and 'coffees' in similar:
                similar_ratings = [c['rating'] for c in similar['coffees']]
                rag_avg = sum(similar_ratings) / len(similar_ratings)
                
                rag_context = {
                    'average_rating': round(rag_avg, 2),
                    'sample_size': len(similar_ratings),
                    'rating_range': [min(similar_ratings), max(similar_ratings)]
                }
                
                # Combine predictions (70% RF, 30% RAG)
                combined_prediction = round(0.7 * result['prediction'] + 0.3 * rag_avg, 2)
        
        return {
            'rf_prediction': round(result['prediction'], 2),
            'rag_context': rag_context,
            'combined_prediction': combined_prediction,
            'explanation': result['explanation'],
            'feature_importance': result['top_3_features']
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-info")
async def get_model_info(ml_predictor = Depends(get_ml_predictor)):
    """Get ML model information"""
    return ml_predictor.get_model_info()