"""
Test Action Predictor Module

This module provides functionality to predict appropriate test actions
for web elements using our trained ML model.
"""

import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Any, Union

class TestActionPredictor:
    """
    Predicts test actions for web elements using trained ML model.
    
    Attributes:
        model: Trained ML model
        scaler: Feature scaler
        feature_columns: List of feature columns used by model
    """
    
    def __init__(
        self,
        model_dir: str = "models/rf",
        model_type: str = "rf"
    ):
        """
        Initialize the predictor with trained model.
        
        Args:
            model_dir: Directory containing trained model files
            model_type: Type of model to load (rf, mlp)
        """
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        
        # Load model and scaler
        self._load_model()
        
        # Define feature columns (must match training)
        self.feature_columns = [
            'text_length', 'has_id', 'has_class', 'num_attributes',
            'position_x', 'position_y', 'element_width', 'element_height',
            'tag_encoded'
        ]
        
        # Set up logging
        logger.add(
            self.model_dir / "prediction.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
    def predict_action(
        self,
        element_data: Dict[str, Any]
    ) -> Dict[str, Union[int, float, str]]:
        """
        Predict test action for a web element.
        
        Args:
            element_data: Dictionary containing element features
            
        Returns:
            Dictionary with predicted action and confidence
        """
        try:
            # Get element tag and attributes
            tag_name = element_data.get('tag_name', '').lower()
            attributes = element_data.get('attributes', {})
            
            # Rule-based predictions for common elements
            if tag_name == 'a':
                return {
                    "action": "click",
                    "confidence": 0.95,
                    "probabilities": {"click": 0.95, "type": 0.03, "submit": 0.02}
                }
            elif tag_name == 'button':
                button_type = attributes.get('type', '')
                if button_type == 'submit':
                    return {
                        "action": "submit",
                        "confidence": 0.95,
                        "probabilities": {"submit": 0.95, "click": 0.04, "type": 0.01}
                    }
                else:
                    return {
                        "action": "click",
                        "confidence": 0.95,
                        "probabilities": {"click": 0.95, "type": 0.03, "submit": 0.02}
                    }
            elif tag_name == 'input':
                input_type = attributes.get('type', 'text')
                if input_type in ['text', 'email', 'password', 'search', 'tel', 'url']:
                    return {
                        "action": "type",
                        "confidence": 0.95,
                        "probabilities": {"type": 0.95, "click": 0.03, "submit": 0.02}
                    }
                elif input_type in ['submit', 'button']:
                    return {
                        "action": "click",
                        "confidence": 0.95,
                        "probabilities": {"click": 0.95, "type": 0.03, "submit": 0.02}
                    }
                elif input_type in ['checkbox', 'radio']:
                    return {
                        "action": "click",
                        "confidence": 0.95,
                        "probabilities": {"click": 0.95, "type": 0.03, "submit": 0.02}
                    }
            
            # Use ML model for other elements
            features = self._prepare_features(element_data)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(max(probabilities))
            
            # Map numeric prediction to action name
            action_map = {
                0: "click",
                1: "type",
                2: "submit"
            }
            
            predicted_action = action_map[prediction]
            
            logger.info(
                f"Predicted action '{predicted_action}' with "
                f"confidence {confidence:.2f} for element: "
                f"{element_data.get('tag_name', 'unknown')}"
            )
            
            return {
                "action": predicted_action,
                "confidence": confidence,
                "probabilities": {
                    action: float(prob)
                    for action, prob in zip(action_map.values(), probabilities)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to predict action: {str(e)}")
            # Default to click with low confidence
            return {
                "action": "click",
                "confidence": 0.5,
                "probabilities": {"click": 0.5, "type": 0.3, "submit": 0.2}
            }
            
    def _prepare_features(
        self,
        element_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Prepare element features for prediction.
        
        Args:
            element_data: Dictionary containing element features
            
        Returns:
            DataFrame with prepared features
        """
        try:
            # Extract features
            features = {
                'text_length': len(element_data.get('inner_text', '')),
                'has_id': 'id' in element_data.get('attributes', {}),
                'has_class': 'class' in element_data.get('attributes', {}),
                'num_attributes': len(element_data.get('attributes', {})),
                'position_x': element_data.get('position', {}).get('x', 0),
                'position_y': element_data.get('position', {}).get('y', 0),
                'element_width': element_data.get('size', {}).get('width', 0),
                'element_height': element_data.get('size', {}).get('height', 0),
                'tag_encoded': self._encode_tag(element_data.get('tag_name', ''))
            }
            
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Scale features
            if hasattr(self, 'scaler'):
                df = pd.DataFrame(
                    self.scaler.transform(df),
                    columns=df.columns
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to prepare features: {str(e)}")
            # Return default features
            return pd.DataFrame([[0] * len(self.feature_columns)],
                              columns=self.feature_columns)
    
    def _encode_tag(self, tag_name: str) -> int:
        """Encode HTML tag to numeric value."""
        tag_map = {
            'a': 1,
            'button': 2,
            'input': 3,
            'select': 4,
            'textarea': 5,
            'form': 6
        }
        return tag_map.get(tag_name.lower(), 0)
        
    def _load_model(self):
        """Load trained model and scaler."""
        try:
            model_path = self.model_dir / f"{self.model_type}_model.joblib"
            scaler_path = self.model_dir / f"{self.model_type}_scaler.joblib"
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            logger.info(f"Loaded model from {model_path}")
            logger.info(f"Loaded scaler from {scaler_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
