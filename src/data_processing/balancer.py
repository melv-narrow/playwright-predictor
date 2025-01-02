"""
Dataset Balancing Module

This module provides various techniques for balancing the training dataset:
1. Random undersampling of majority class
2. Random oversampling of minority classes
3. SMOTE (Synthetic Minority Over-sampling Technique)
4. Combination approaches

The goal is to improve model performance on underrepresented element types
and actions.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from loguru import logger

class DataBalancer:
    """
    A class for balancing datasets using various techniques.
    
    Attributes:
        strategy (str): Balancing strategy to use
        target_ratios (Dict[str, float]): Target ratios for each class
        random_state (int): Random seed for reproducibility
    """
    
    VALID_STRATEGIES = ['undersample', 'oversample', 'smote', 'combined']
    
    def __init__(
        self,
        strategy: str = 'combined',
        target_ratios: Dict[str, float] = None,
        random_state: int = 42
    ):
        """
        Initialize the data balancer.
        
        Args:
            strategy: Balancing strategy ('undersample', 'oversample', 'smote', 'combined')
            target_ratios: Target ratios for each class (e.g., {'click': 0.4, 'type': 0.3})
            random_state: Random seed for reproducibility
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Strategy must be one of {self.VALID_STRATEGIES}")
            
        self.strategy = strategy
        self.target_ratios = target_ratios or {}
        self.random_state = random_state
        
    def balance_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: list
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance the dataset using the specified strategy.
        
        Args:
            X: Feature DataFrame
            y: Target labels
            feature_columns: Columns to use for SMOTE/sampling
            
        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        try:
            logger.info(f"Balancing dataset using {self.strategy} strategy")
            logger.info(f"Original class distribution:\n{y.value_counts()}")
            
            # Prepare feature matrix for sampling
            X_features = X[feature_columns].copy()
            
            # Handle non-numeric features
            scaler = StandardScaler()
            X_features = pd.DataFrame(
                scaler.fit_transform(X_features),
                columns=feature_columns
            )
            
            # Get class distribution
            class_counts = y.value_counts()
            min_samples = class_counts.min()
            
            # Determine sampling strategy based on class distribution
            if self.strategy in ['smote', 'combined']:
                # For SMOTE, ensure k_neighbors is less than min_samples
                k_neighbors = min(3, min_samples - 1)
                logger.info(f"Using k_neighbors={k_neighbors} for SMOTE")
                
                # For very small classes, fall back to oversampling
                if k_neighbors < 2:
                    logger.warning("Too few samples for SMOTE, falling back to oversampling")
                    self.strategy = 'oversample'
            
            # Apply balancing strategy
            if self.strategy == 'undersample':
                X_resampled, y_resampled = self._undersample(X_features, y)
            elif self.strategy == 'oversample':
                X_resampled, y_resampled = self._oversample(X_features, y)
            elif self.strategy == 'smote':
                X_resampled, y_resampled = self._apply_smote(X_features, y, k_neighbors)
            else:  # combined
                X_resampled, y_resampled = self._combined_approach(X_features, y, k_neighbors)
                
            # Reconstruct full DataFrame with balanced data
            X_balanced = X.copy()
            X_balanced[feature_columns] = pd.DataFrame(
                scaler.inverse_transform(X_resampled),
                columns=feature_columns,
                index=X_resampled.index
            )
            
            logger.info(f"New class distribution:\n{y_resampled.value_counts()}")
            return X_balanced, y_resampled
            
        except Exception as e:
            logger.error(f"Dataset balancing failed: {str(e)}")
            raise
            
    def _undersample(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply random undersampling to majority class."""
        sampling_strategy = self.target_ratios or 'auto'
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        return sampler.fit_resample(X, y)
        
    def _oversample(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply random oversampling to minority classes."""
        sampling_strategy = self.target_ratios or 'auto'
        sampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        return sampler.fit_resample(X, y)
        
    def _apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k_neighbors: int = 5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE to generate synthetic samples."""
        sampling_strategy = self.target_ratios or 'auto'
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=k_neighbors
        )
        return smote.fit_resample(X, y)
        
    def _combined_approach(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k_neighbors: int = 5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply a combination of SMOTE and undersampling.
        This often provides better results than either technique alone.
        """
        sampling_strategy = self.target_ratios or 'auto'
        
        # Create a pipeline of SMOTE followed by undersampling
        pipeline = Pipeline([
            ('smote', SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=k_neighbors
            )),
            ('undersampler', RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            ))
        ])
        
        return pipeline.fit_resample(X, y)