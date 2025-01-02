"""
ML Model Module

This module handles the training and evaluation of ML models for predicting
appropriate test actions based on web element features.

Key Features:
- Multiple model architectures (Random Forest, XGBoost, Neural Network)
- Feature importance analysis
- Model evaluation and comparison
- Cross-validation
- Hyperparameter tuning
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from loguru import logger

class ActionPredictor:
    """
    A class for training and using ML models to predict test actions.
    
    This class handles:
    - Model training and evaluation
    - Feature preprocessing
    - Model selection
    - Prediction generation
    
    Attributes:
        model_type (str): Type of model to use
        model: Trained model instance
        scaler: Feature scaler
        feature_columns (list): Columns to use as features
    """
    
    VALID_MODELS = ['rf', 'xgb', 'mlp']
    
    def __init__(
        self,
        model_type: str = 'rf',
        model_dir: str = 'models',
        random_state: int = 42
    ):
        """
        Initialize the action predictor.
        
        Args:
            model_type: Type of model ('rf', 'xgb', 'mlp')
            model_dir: Directory to save models
            random_state: Random seed for reproducibility
        """
        if model_type not in self.VALID_MODELS:
            raise ValueError(f"Model type must be one of {self.VALID_MODELS}")
            
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.random_state = random_state
        
        # Create model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        
        # Set up logging
        logger.add(
            self.model_dir / "training.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
        # Define feature columns
        self.feature_columns = [
            'text_length', 'has_id', 'has_class', 'num_attributes',
            'position_x', 'position_y', 'element_width', 'element_height',
            'tag_encoded'
        ]
        
    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: Optional[pd.DataFrame] = None,
        tune_hyperparams: bool = True
    ) -> Dict[str, float]:
        """
        Train the model on processed data.
        
        Args:
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            logger.info(f"Training {self.model_type} model")
            
            # Prepare features
            X_train = self._prepare_features(train_data)
            y_train = train_data['action_label']
            
            if valid_data is not None:
                X_valid = self._prepare_features(valid_data)
                y_valid = valid_data['action_label']
            
            # Initialize model
            if tune_hyperparams:
                self.model = self._train_with_tuning(X_train, y_train)
            else:
                self.model = self._train_base_model(X_train, y_train)
            
            # Evaluate model
            metrics = {}
            
            # Training metrics
            train_preds = self.model.predict(X_train)
            metrics['train'] = classification_report(
                y_train,
                train_preds,
                output_dict=True
            )
            
            # Validation metrics
            if valid_data is not None:
                valid_preds = self.model.predict(X_valid)
                metrics['valid'] = classification_report(
                    y_valid,
                    valid_preds,
                    output_dict=True
                )
            
            # Cross-validation score
            cv_scores = cross_val_score(
                self.model,
                X_train,
                y_train,
                cv=5,
                scoring='f1_weighted'
            )
            metrics['cv_score'] = cv_scores.mean()
            
            # Log results
            self._log_training_results(metrics)
            
            # Save model
            self._save_model()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
            
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new data.
        
        Args:
            data: DataFrame containing features
            
        Returns:
            Array of predicted actions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        X = self._prepare_features(data)
        return self.model.predict(X)
        
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input."""
        # Select and scale features
        X = data[self.feature_columns].copy()
        
        # Scale numerical features if not already scaled
        numerical_features = [
            'text_length', 'num_attributes',
            'position_x', 'position_y',
            'element_width', 'element_height'
        ]
        
        if not hasattr(self, 'scaler_fitted_'):
            self.scaler.fit(X[numerical_features])
            self.scaler_fitted_ = True
            
        X[numerical_features] = self.scaler.transform(X[numerical_features])
        
        return X
        
    def _train_base_model(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Any:
        """Train model with default parameters."""
        if self.model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=self.random_state
            )
        elif self.model_type == 'xgb':
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state
            )
        else:  # mlp
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state
            )
            
        model.fit(X, y)
        return model
        
    def _train_with_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Any:
        """Train model with hyperparameter tuning."""
        # Define parameter grid based on model type
        if self.model_type == 'rf':
            model = RandomForestClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'xgb':
            model = XGBClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        else:  # mlp
            model = MLPClassifier(random_state=self.random_state)
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
            
        # Perform grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
        
    def _log_training_results(self, metrics: Dict[str, Any]):
        """Log training results and metrics."""
        logger.info("\n=== Training Results ===")
        
        # Training metrics
        logger.info("\nTraining Metrics:")
        for metric, value in metrics['train'].items():
            if isinstance(value, dict):
                logger.info(f"\n{metric}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"{metric}: {value:.4f}")
        
        # Validation metrics if available
        if 'valid' in metrics:
            logger.info("\nValidation Metrics:")
            for metric, value in metrics['valid'].items():
                if isinstance(value, dict):
                    logger.info(f"\n{metric}:")
                    for k, v in value.items():
                        logger.info(f"  {k}: {v:.4f}")
                else:
                    logger.info(f"{metric}: {value:.4f}")
        
        # Cross-validation score
        logger.info(f"\nCross-validation F1 Score: {metrics['cv_score']:.4f}")
        
    def _save_model(self):
        """Save trained model and scaler."""
        model_path = self.model_dir / f"{self.model_type}_model.joblib"
        scaler_path = self.model_dir / f"{self.model_type}_scaler.joblib"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        
    def load_model(self, model_path: str, scaler_path: str):
        """Load trained model and scaler from files."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.scaler_fitted_ = True
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Loaded scaler from {scaler_path}")
