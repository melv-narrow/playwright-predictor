#!/usr/bin/env python
"""
Model Training Script

This script trains and evaluates different ML models on our processed dataset.
It compares Random Forest, XGBoost, and Neural Network models to find the
best performer for our test action prediction task.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from loguru import logger
from src.ml.model import ActionPredictor

def train_and_evaluate(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_type: str,
    tune_hyperparams: bool = True
):
    """Train and evaluate a specific model type."""
    try:
        logger.info(f"\n=== Training {model_type.upper()} Model ===")
        
        # Initialize model
        predictor = ActionPredictor(
            model_type=model_type,
            model_dir=f"models/{model_type}",
            random_state=42
        )
        
        # Train model
        metrics = predictor.train(
            train_data=train_data,
            valid_data=test_data,
            tune_hyperparams=tune_hyperparams
        )
        
        return metrics, predictor
        
    except Exception as e:
        logger.error(f"Training failed for {model_type}: {str(e)}")
        return None, None

def main():
    """Main execution function."""
    try:
        # Load processed data
        data_dir = Path("training_data/automation_exercise/processed_smote")
        train_data = pd.read_pickle(data_dir / "train.pkl")
        test_data = pd.read_pickle(data_dir / "test.pkl")
        
        logger.info(f"Loaded {len(train_data)} training samples")
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Train all model types
        models = ['rf', 'xgb', 'mlp']
        results = {}
        
        for model_type in models:
            metrics, predictor = train_and_evaluate(
                train_data=train_data,
                test_data=test_data,
                model_type=model_type,
                tune_hyperparams=True
            )
            
            if metrics:
                results[model_type] = metrics
        
        # Compare results
        logger.info("\n=== Model Comparison ===")
        for model_type, metrics in results.items():
            logger.info(f"\n{model_type.upper()} Results:")
            logger.info(f"CV Score: {metrics['cv_score']:.4f}")
            logger.info(f"Validation F1-weighted: {metrics['valid']['weighted avg']['f1-score']:.4f}")
        
    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging
    log_path = Path("models/training.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, rotation="1 day", retention="7 days", level="INFO")
    
    # Run training
    main()
