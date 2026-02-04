"""
RAG Engine for Coffee Recommendations
"""
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CoffeeRAGEngine:
    """RAG Engine for coffee brewing recommendations"""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self.collection = None
        self.df = None
        
    async def initialize(self):
        """Initialize ChromaDB and load data"""
        try:
            logger.info("Initializing RAG Engine...")
            
            # Initialize ChromaDB
            self.client = chromadb.PersistentClient(
                path=str(self.config.CHROMA_DIR)
            )
            
            # Create embedding function
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.EMBEDDING_MODEL
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.COLLECTION_NAME,
                    embedding_function=embedding_fn
                )
                logger.info(f"✓ Loaded existing collection: {self.config.COLLECTION_NAME}")
            except:
                self.collection = self.client.create_collection(
                    name=self.config.COLLECTION_NAME,
                    embedding_function=embedding_fn
                )
                logger.info(f"✓ Created new collection: {self.config.COLLECTION_NAME}")
            
            # Load dataset
            self.df = pd.read_csv(self.config.DATASET_PATH)
            logger.info(f"✓ Loaded {len(self.df)} records")
            
            # Populate if empty
            if self.collection.count() == 0:
                logger.info("Populating database...")
                await self._populate_database()
            else:
                logger.info(f"✓ Collection has {self.collection.count()} records")
            
            logger.info("✓ RAG Engine initialized!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise
    
    def _create_coffee_description(self, row: pd.Series) -> str:
        """Create text description for a coffee"""
        
        detected_flavors = []
        flavor_cols = ['Sweet', 'Nutty', 'Dark.chocolate', 'Caramel', 'Fruit', 
                       'Citrus', 'Bitter', 'Sour', 'Burnt', 'Rubber', 'Roasted']
        
        for col in flavor_cols:
            if col in row and row[col] == 1:
                detected_flavors.append(col.replace('.', ' ').lower())
        
        description = f"""
        Coffee rated {row['Liking']}/9 by Judge {row['Judge']} (Cluster {row['Cluster']}).
        Brewing: {row['Brew Temperature']:.0f}°C, TDS {row['TDS__1']:.2f}%, Extraction {row['Percent Extraction']:.0f}%.
        Flavor intensity: {row['Flavor.intensity']}/5, Acidity: {row['Acidity']}/5, Mouthfeel: {row['Mouthfeel']}/5.
        Detected flavors: {', '.join(detected_flavors) if detected_flavors else 'none'}.
        """
        
        return description.strip()
    
    async def _populate_database(self):
        """Populate vector database"""
        
        batch_size = 100
        total = 0
        
        for i in range(0, len(self.df), batch_size):
            batch = self.df.iloc[i:i+batch_size]
            
            ids = [f"coffee_{idx}" for idx in batch.index]
            documents = [self._create_coffee_description(row) for _, row in batch.iterrows()]
            
            metadatas = []
            for _, row in batch.iterrows():
                metadata = {
                    'liking': float(row['Liking']),
                    'judge': int(row['Judge']),
                    'cluster': int(row['Cluster']),
                    'brew_temp': float(row['Brew Temperature']) if pd.notna(row['Brew Temperature']) else 0,
                    'tds': float(row['TDS__1']) if pd.notna(row['TDS__1']) else 0,
                    'extraction': float(row['Percent Extraction']) if pd.notna(row['Percent Extraction']) else 0,
                    'dose': float(row['Dose']) if pd.notna(row['Dose']) else 0,
                    'ph': float(row['pH']) if pd.notna(row['pH']) else 0,
                    'sweet': int(row.get('Sweet', 0)) if pd.notna(row.get('Sweet', 0)) else 0,
                    'bitter': int(row.get('Bitter', 0)) if pd.notna(row.get('Bitter', 0)) else 0,
                    'dark_chocolate': int(row.get('Dark.chocolate', 0)) if pd.notna(row.get('Dark.chocolate', 0)) else 0,
                    'caramel': int(row.get('Caramel', 0)) if pd.notna(row.get('Caramel', 0)) else 0,
                    'nutty': int(row.get('Nutty', 0)) if pd.notna(row.get('Nutty', 0)) else 0,
                    'sour': int(row.get('Sour', 0)) if pd.notna(row.get('Sour', 0)) else 0,
                    'fruit': int(row.get('Fruit', 0)) if pd.notna(row.get('Fruit', 0)) else 0,
                    'roasted': int(row.get('Roasted', 0)) if pd.notna(row.get('Roasted', 0)) else 0,
                }
                metadatas.append(metadata)
            
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            total += len(batch)
            
            if total % 500 == 0:
                logger.info(f"  Added {total}/{len(self.df)} records...")
        
        logger.info(f"✓ Populated {total} records")
    
    def get_brewing_recipe(
        self,
        user_preference: str,
        n_results: int = 50,
        filter_type: Optional[str] = None,
        filter_value: Optional[float | int] = None
    ) -> Dict:
        """Get brewing recipe based on preference"""
        
        try:
            where_filter = None

            if filter_type == "min_rating":
                where_filter = {"liking": {"$gte": float(filter_value)}}

            elif filter_type == "cluster":
                where_filter = {"cluster": int(filter_value)}

            
            results = self.collection.query(
                query_texts=[user_preference],
                n_results=n_results,
                where=where_filter
            )
            
            if not results['metadatas'][0]:
                return {'error': 'No matching coffees found'}
            
            coffees = results['metadatas'][0]
            
            # Calculate statistics
            temps = [c['brew_temp'] for c in coffees if c['brew_temp'] > 0]
            tds_vals = [c['tds'] for c in coffees if c['tds'] > 0]
            extractions = [c['extraction'] for c in coffees if c['extraction'] > 0]
            doses = [c['dose'] for c in coffees if c['dose'] > 0]
            ratings = [c['liking'] for c in coffees]
            
            # Sensory profile
            sensory_stats = {
                'sweet': np.mean([c['sweet'] for c in coffees]) * 100,
                'bitter': np.mean([c['bitter'] for c in coffees]) * 100,
                'dark_chocolate': np.mean([c['dark_chocolate'] for c in coffees]) * 100,
                'caramel': np.mean([c['caramel'] for c in coffees]) * 100,
                'nutty': np.mean([c['nutty'] for c in coffees]) * 100,
                'sour': np.mean([c['sour'] for c in coffees]) * 100,
                'fruit': np.mean([c['fruit'] for c in coffees]) * 100,
                'roasted': np.mean([c['roasted'] for c in coffees]) * 100,
            }
            
            sensory_sorted = {k: round(v, 1) for k, v in sorted(
                sensory_stats.items(), key=lambda x: x[1], reverse=True
            ) if v > 10}
            
            top_coffees = sorted(coffees, key=lambda x: x['liking'], reverse=True)[:5]
            
            return {
                'query': user_preference,
                'sample_size': len(coffees),
                'brewing_parameters': {
                    'temperature': {
                        'optimal': round(float(np.mean(temps)), 1) if temps else None,
                        'range': [round(float(np.min(temps)), 1), round(float(np.max(temps)), 1)] if temps else None,
                        'sweet_spot': [round(float(np.percentile(temps, 25)), 1), round(float(np.percentile(temps, 75)), 1)] if temps else None
                    },
                    'tds': {
                        'optimal': round(float(np.mean(tds_vals)), 2) if tds_vals else None,
                        'range': [round(float(np.min(tds_vals)), 2), round(float(np.max(tds_vals)), 2)] if tds_vals else None
                    },
                    'extraction': {
                        'optimal': round(float(np.mean(extractions)), 1) if extractions else None,
                        'range': [round(float(np.min(extractions)), 1), round(float(np.max(extractions)), 1)] if extractions else None
                    },
                    'dose': {
                        'optimal': round(float(np.mean(doses)), 1) if doses else None,
                        'range': [round(float(np.min(doses)), 1), round(float(np.max(doses)), 1)] if doses else None
                    }
                },
                'expected_results': {
                    'average_rating': round(float(np.mean(ratings)), 2),
                    'rating_range': [round(float(np.min(ratings)), 1), round(float(np.max(ratings)), 1)],
                    'most_common_rating': round(float(max(set(ratings), key=ratings.count)), 1)
                },
                'sensory_profile': sensory_sorted,
                'top_examples': [
                    {
                        'judge': c['judge'],
                        'rating': c['liking'],
                        'temp': c['brew_temp'],
                        'tds': c['tds'],
                        'extraction': c['extraction']
                    }
                    for c in top_coffees
                ],
                'brewing_tips': self._generate_tips(temps, extractions, sensory_stats)
            }
            
        except Exception as e:
            logger.error(f"Error in get_brewing_recipe: {e}")
            return {'error': str(e)}
    
    def _generate_tips(self, temps, extractions, sensory):
        """Generate brewing tips"""
        tips = []
        
        if temps and np.mean(temps) < 89:
            tips.append("Lower temperature preserves sweetness and reduces bitterness")
        
        if extractions and np.mean(extractions) < 21:
            tips.append("Moderate extraction avoids over-extraction and bitterness")
        
        if sensory.get('sweet', 0) > 50:
            tips.append("This profile reliably produces sweet notes")
        
        if sensory.get('bitter', 0) < 20:
            tips.append("Low bitterness - good for bitter-sensitive drinkers")
        
        return tips
    
    def find_similar_coffees(self, query: str, n_results: int = 10) -> Dict:
        """Find similar coffees"""
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            coffees = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                coffees.append({
                    'description': doc,
                    'rating': meta['liking'],
                    'judge': meta['judge'],
                    'cluster': meta['cluster'],
                    'brewing': {
                        'temperature': meta['brew_temp'],
                        'tds': meta['tds'],
                        'extraction': meta['extraction'],
                        'dose': meta['dose']
                    },
                    'flavors': {
                        'sweet': bool(meta['sweet']),
                        'bitter': bool(meta['bitter']),
                        'chocolate': bool(meta['dark_chocolate']),
                        'caramel': bool(meta['caramel']),
                        'nutty': bool(meta['nutty'])
                    }
                })
            
            return {
                'query': query,
                'count': len(coffees),
                'coffees': coffees
            }
            
        except Exception as e:
            logger.error(f"Error in find_similar_coffees: {e}")
            return {'error': str(e)}
    
    def compare_preferences(self, pref1: str, pref2: str) -> Dict:
        """Compare two preferences"""
        
        try:
            recipe1 = self.get_brewing_recipe(pref1)
            recipe2 = self.get_brewing_recipe(pref2)
            
            if 'error' in recipe1 or 'error' in recipe2:
                return {'error': 'Could not retrieve recipes'}
            
            temp_diff = (recipe2['brewing_parameters']['temperature']['optimal'] - 
                        recipe1['brewing_parameters']['temperature']['optimal'])
            
            tds_diff = (recipe2['brewing_parameters']['tds']['optimal'] - 
                       recipe1['brewing_parameters']['tds']['optimal'])
            
            extraction_diff = (recipe2['brewing_parameters']['extraction']['optimal'] - 
                              recipe1['brewing_parameters']['extraction']['optimal'])
            
            return {
                'preference_1': {'query': pref1, 'recipe': recipe1},
                'preference_2': {'query': pref2, 'recipe': recipe2},
                'differences': {
                    'temperature': round(temp_diff, 1),
                    'tds': round(tds_diff, 2),
                    'extraction': round(extraction_diff, 1)
                }
            }
            
        except Exception as e:
            return {'error': str(e)}