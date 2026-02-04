"""
Machine Learning Predictor - Random Forest inference
"""
import joblib
import json
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MLPredictor:
    """ML Model predictor with optimized loading"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        
    def load_models(self):
        """Load pre-trained models"""
        try:
            logger.info("Loading ML models...")
            
            # Load Random Forest
            self.model = joblib.load(self.config.RF_MODEL_PATH)
            logger.info(f"✓ Loaded Random Forest model")
            
            # Load Scaler
            self.scaler = joblib.load(self.config.SCALER_PATH)
            logger.info(f"✓ Loaded StandardScaler")
            
            # Load metadata
            with open(self.config.METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['feature_names']
            logger.info(f"✓ Loaded metadata ({len(self.feature_names)} features)")
            
            # Verify model
            logger.info(f"Model performance - R²: {self.metadata['performance']['r2_score']:.4f}, "
                       f"RMSE: {self.metadata['performance']['rmse']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            raise
    
    def predict(self, sensory_features: list) -> dict:
        """
        Predict liking score from sensory features
        
        Args:
            sensory_features: List of 20 sensory feature values
        
        Returns:
            Dictionary with prediction and feature importance
        """
        try:
            # Validate input
            if len(sensory_features) != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {len(sensory_features)}")
            
            # Convert to numpy array
            features = np.array(sensory_features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            
            # Get feature importance for this prediction
            feature_importance = {
                name: float(imp) 
                for name, imp in zip(self.feature_names, self.model.feature_importances_)
            }
            
            # Sort by importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            return {
                'prediction': float(prediction),
                'feature_importance': sorted_importance,
                'top_3_features': dict(list(sorted_importance.items())[:3])
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_with_explanation(self, sensory_features: list) -> dict:
        """Predict with detailed explanation"""
        
        result = self.predict(sensory_features)
        
        # Generate explanation
        prediction = result['prediction']
        top_features = result['top_3_features']
        
        # Create human-readable explanation
        explanation_parts = []
        
        # Overall prediction
        if prediction >= 7.5:
            explanation_parts.append(f"Predicted rating: {prediction:.1f}/9 - Highly likely to be enjoyed!")
        elif prediction >= 6.5:
            explanation_parts.append(f"Predicted rating: {prediction:.1f}/9 - Good coffee with solid appeal")
        elif prediction >= 5.5:
            explanation_parts.append(f"Predicted rating: {prediction:.1f}/9 - Moderate appeal")
        else:
            explanation_parts.append(f"Predicted rating: {prediction:.1f}/9 - May not be widely liked")
        
        # Top features
        top_feature_names = list(top_features.keys())
        explanation_parts.append(
            f"Main drivers: {', '.join(top_feature_names[:3])} "
            f"(account for {sum(top_features.values()):.1%} of prediction)"
        )
        
        # Specific feature insights
        feature_values = dict(zip(self.feature_names, sensory_features))
        
        if feature_values.get('Sweet', 0) == 1:
            explanation_parts.append("Sweet notes detected - typically increases liking")
        
        if feature_values.get('Bitter', 0) == 1:
            explanation_parts.append("Bitter notes detected - may reduce liking for some drinkers")
        
        result['explanation'] = '. '.join(explanation_parts)
        
        return result
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_type': self.metadata['model_type'],
            'n_features': self.metadata['n_features'],
            'performance': self.metadata['performance'],
            'hyperparameters': self.metadata['hyperparameters'],
            'training_samples': self.metadata['training_samples']
        }