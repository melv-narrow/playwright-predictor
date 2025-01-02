"""
Data Processor Module

This module processes raw website data into features suitable for ML model training.
It handles cleaning, normalization, feature extraction, and dataset preparation.

Key Features:
- HTML cleaning and normalization
- Text feature extraction
- Numerical feature scaling
- Categorical encoding
- Feature vector generation
- Dataset splitting and validation

Usage Example:
    processor = DataProcessor(input_dir="raw_data", output_dir="processed_data")
    processor.process_dataset()
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger
import re
from .balancer import DataBalancer

class DataProcessor:
    """
    A class for processing raw website data into ML-ready features.
    
    This class handles all aspects of data processing including:
    - Data cleaning and normalization
    - Feature extraction and engineering
    - Dataset preparation and splitting
    
    Attributes:
        input_dir (Path): Directory containing raw data
        output_dir (Path): Directory to save processed data
        label_encoder (LabelEncoder): Encoder for action labels
        scaler (StandardScaler): Scaler for numerical features
        balance_strategy (str): Strategy for balancing dataset
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        test_size: float = 0.2,
        random_state: int = 42,
        balance_strategy: str = 'combined'
    ):
        """
        Initialize the data processor.
        
        Args:
            input_dir: Directory containing raw data files
            output_dir: Directory to save processed data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            balance_strategy: Strategy for balancing dataset
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.balance_strategy = balance_strategy
        
        # Initialize encoders and scalers
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Initialize balancer
        self.balancer = DataBalancer(
            strategy=balance_strategy,
            random_state=random_state
        )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logger.add(
            self.output_dir / "processor.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
    def process_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the entire dataset from raw data to ML-ready features.
        
        Returns:
            Tuple of (train_df, test_df) containing processed features
        """
        try:
            # Load and combine raw data
            raw_data = self._load_raw_data()
            if not raw_data:
                raise ValueError("No valid data files found")
            
            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            logger.info(f"Loaded {len(df)} raw data samples")
            
            # Process features
            processed_df = self._process_features(df)
            
            # Get numerical feature columns for balancing
            numerical_features = [
                'text_length', 'num_attributes',
                'position_x', 'position_y',
                'element_width', 'element_height'
            ]
            
            # Split dataset
            train_df, test_df = train_test_split(
                processed_df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=processed_df['action_label']
            )
            
            # Balance training data
            train_df, train_df['action_label'] = self.balancer.balance_dataset(
                train_df,
                train_df['action_label'],
                numerical_features
            )
            
            # Save processed datasets
            self._save_processed_data(train_df, test_df)
            
            logger.info(f"Processing complete. Train size: {len(train_df)}, Test size: {len(test_df)}")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {str(e)}")
            raise
            
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load and combine all raw data files.
        
        Returns:
            List of dictionaries containing raw data
        """
        all_data = []
        
        try:
            # Load all JSON files
            json_files = list(self.input_dir.glob("raw_data_*.json"))
            for file_path in json_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {str(e)}")
                    continue
            
            logger.info(f"Loaded {len(all_data)} samples from {len(json_files)} files")
            return all_data
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {str(e)}")
            raise
            
    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw features into ML-ready format.
        
        This method:
        1. Cleans and normalizes HTML
        2. Extracts text features
        3. Processes numerical features
        4. Encodes categorical features
        
        Args:
            df: DataFrame containing raw features
            
        Returns:
            DataFrame with processed features
        """
        try:
            # Create feature columns
            df['clean_html'] = df['html'].apply(self._clean_html)
            df['text_length'] = df['inner_text'].str.len()
            df['has_id'] = df['attributes'].apply(lambda x: 'id' in x)
            df['has_class'] = df['attributes'].apply(lambda x: 'class' in x)
            df['num_attributes'] = df['attributes'].apply(len)
            df['position_x'] = df['bounding_box'].apply(lambda x: x.get('x', 0) if x else 0)
            df['position_y'] = df['bounding_box'].apply(lambda x: x.get('y', 0) if x else 0)
            df['element_width'] = df['bounding_box'].apply(lambda x: x.get('width', 0) if x else 0)
            df['element_height'] = df['bounding_box'].apply(lambda x: x.get('height', 0) if x else 0)
            
            # Extract text features
            df['text_features'] = df.apply(self._extract_text_features, axis=1)
            
            # Process numerical features
            numerical_features = [
                'text_length', 'num_attributes',
                'position_x', 'position_y',
                'element_width', 'element_height'
            ]
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
            
            # Encode categorical features
            df['tag_encoded'] = self.label_encoder.fit_transform(df['tag_name'])
            df['action_label'] = self.label_encoder.fit_transform(df['suggested_action'])
            
            # Save encoders and scalers
            self._save_preprocessors()
            
            logger.info("Feature processing complete")
            return df
            
        except Exception as e:
            logger.error(f"Feature processing failed: {str(e)}")
            raise
            
    def _clean_html(self, html: str) -> str:
        """
        Clean and normalize HTML content.
        
        Args:
            html: Raw HTML string
            
        Returns:
            Cleaned HTML string
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, str)):
                comment.extract()
            
            # Normalize whitespace
            text = soup.get_text(separator=' ')
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception:
            return html
            
    def _extract_text_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Extract features from element text content.
        
        Args:
            row: DataFrame row containing element data
            
        Returns:
            Dictionary of extracted features
        """
        text = str(row['inner_text'])
        clean_html = str(row['clean_html'])
        
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_number': bool(re.search(r'\d', text)),
            'has_special_chars': bool(re.search(r'[^\w\s]', text)),
            'html_length': len(clean_html),
            'html_word_count': len(clean_html.split())
        }
        
        return features
        
    def _save_preprocessors(self):
        """Save fitted preprocessors for later use."""
        try:
            # Save label encoder
            with open(self.output_dir / 'label_encoder.json', 'w') as f:
                json.dump({
                    'classes_': self.label_encoder.classes_.tolist()
                }, f)
            
            # Save scaler parameters
            with open(self.output_dir / 'scaler.json', 'w') as f:
                json.dump({
                    'mean_': self.scaler.mean_.tolist(),
                    'scale_': self.scaler.scale_.tolist()
                }, f)
                
            logger.info("Saved preprocessors")
            
        except Exception as e:
            logger.error(f"Failed to save preprocessors: {str(e)}")
            
    def _save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Save processed datasets to disk.
        
        Args:
            train_df: Training dataset
            test_df: Testing dataset
        """
        try:
            # Save as CSV
            train_df.to_csv(self.output_dir / 'train.csv', index=False)
            test_df.to_csv(self.output_dir / 'test.csv', index=False)
            
            # Save as pickle for preserving data types
            train_df.to_pickle(self.output_dir / 'train.pkl')
            test_df.to_pickle(self.output_dir / 'test.pkl')
            
            logger.info("Saved processed datasets")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
            raise
