"""
Analysis router - Statistics and insights
"""
from fastapi import APIRouter, HTTPException, Depends, Path
from app.main import get_rag_engine
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["Analysis"])

@router.get("/cluster/{cluster_id}")
async def get_cluster_preferences(
    cluster_id: int = Path(..., ge=1, le=2),
    rag_engine = Depends(get_rag_engine)
):
    """Get preferences for a specific cluster"""
    
    # Get all data for cluster
    try:
        results = rag_engine.collection.get(
            where={"cluster": cluster_id},
            limit=1000
        )
        
        if not results['metadatas']:
            raise HTTPException(status_code=404, detail=f"No data for cluster {cluster_id}")
        
        coffees = results['metadatas']
        high_rated = [c for c in coffees if c['liking'] >= 7]
        
        if not high_rated:
            raise HTTPException(status_code=404, detail=f"No highly-rated coffees in cluster {cluster_id}")
        
        import numpy as np
        
        sensory_prefs = {
            'sweet': np.mean([c['sweet'] for c in high_rated]) * 100,
            'bitter': np.mean([c['bitter'] for c in high_rated]) * 100,
            'dark_chocolate': np.mean([c['dark_chocolate'] for c in high_rated]) * 100,
        }
        
        temps = [c['brew_temp'] for c in high_rated if c['brew_temp'] > 0]
        
        return {
            'cluster': cluster_id,
            'sample_size': len(high_rated),
            'taste_profile': sensory_prefs,
            'optimal_temperature': round(np.mean(temps), 1) if temps else None,
            'average_rating': round(np.mean([c['liking'] for c in high_rated]), 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_database_stats(rag_engine = Depends(get_rag_engine)):
    """Get database statistics"""
    
    try:
        total = rag_engine.collection.count()
        sample = rag_engine.collection.get(limit=500)
        
        if not sample['metadatas']:
            raise HTTPException(status_code=404, detail="No data in database")
        
        import numpy as np
        coffees = sample['metadatas']
        
        return {
            'total_records': total,
            'avg_rating': round(np.mean([c['liking'] for c in coffees]), 2),
            'cluster_distribution': {
                'cluster_1': len([c for c in coffees if c['cluster'] == 1]),
                'cluster_2': len([c for c in coffees if c['cluster'] == 2])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))