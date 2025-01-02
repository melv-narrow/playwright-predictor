#!/usr/bin/env python
"""
Training Data Processing Script

This script processes the raw data collected from websites into a format
suitable for training ML models. It handles data cleaning, feature extraction,
and dataset splitting.

The script will:
1. Load raw data from JSON files
2. Clean and normalize the data
3. Extract relevant features
4. Balance the dataset
5. Split into training and testing sets
6. Save processed datasets
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from loguru import logger
from src.data_processing.processor import DataProcessor

def process_with_strategy(strategy: str):
    """Process dataset with a specific balancing strategy."""
    try:
        # Set up paths
        base_dir = Path("training_data/automation_exercise")
        input_dir = base_dir
        output_dir = base_dir / f"processed_{strategy}"
        
        # Initialize processor
        processor = DataProcessor(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            test_size=0.2,  # 80% training, 20% testing
            random_state=42,  # For reproducibility
            balance_strategy=strategy
        )
        
        # Process data
        logger.info(f"\n=== Processing with {strategy} strategy ===")
        train_df, test_df = processor.process_dataset()
        
        # Log statistics
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Testing samples: {len(test_df)}")
        logger.info(f"Feature columns: {list(train_df.columns)}")
        logger.info(f"Action distribution:\n{train_df['action_label'].value_counts()}")
        logger.info(f"Element distribution:\n{train_df['tag_name'].value_counts()}")
        
    except Exception as e:
        logger.error(f"Processing failed for {strategy}: {str(e)}")

def main():
    """Main execution function."""
    try:
        # Try different balancing strategies
        strategies = ['undersample', 'oversample', 'smote', 'combined']
        
        for strategy in strategies:
            process_with_strategy(strategy)
            
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging
    log_path = Path("training_data/automation_exercise/process.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, rotation="1 day", retention="7 days", level="INFO")
    
    # Run processing
    main()
