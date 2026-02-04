"""
Recipe router - Brewing recommendations
"""
from fastapi import APIRouter, HTTPException, Depends
from app.schemas import RecipeRequest, RecipeResponse, SimilarCoffeesRequest, CompareRequest
from app.main import get_rag_engine
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recipe", tags=["Brewing Recipes"])

@router.post("/", response_model=RecipeResponse)
async def get_brewing_recipe(request: RecipeRequest, rag_engine = Depends(get_rag_engine)):
    """Get brewing recipe based on user preference"""
    
    result = rag_engine.get_brewing_recipe(
        user_preference=request.preference,
        n_results=request.n_results,
        filter_type=request.filter_type,
        filter_value=request.filter_value
    )

    
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    
    return result

@router.post("/similar")
async def find_similar_coffees(request: SimilarCoffeesRequest, rag_engine = Depends(get_rag_engine)):
    """Find similar coffees based on query"""
    
    result = rag_engine.find_similar_coffees(request.query, request.n_results)
    
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    
    return result

@router.post("/compare")
async def compare_preferences(request: CompareRequest, rag_engine = Depends(get_rag_engine)):
    """Compare two coffee preferences"""
    
    result = rag_engine.compare_preferences(request.preference_1, request.preference_2)
    
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    
    return result