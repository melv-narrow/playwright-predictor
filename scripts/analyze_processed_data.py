"""
Analyze the processed training data to understand feature distributions
and relationships between elements and actions.
"""

import pandas as pd
import json
from pathlib import Path

def main():
    # Load processed data
    train_df = pd.read_csv('training_data/automation_exercise/processed/train.csv')
    
    # Print basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Total samples: {len(train_df)}")
    
    # Analyze action distribution
    print("\n=== Action Distribution ===")
    action_dist = pd.DataFrame({
        'tag_name': train_df['tag_name'],
        'suggested_action': train_df['suggested_action'],
        'count': 1
    }).groupby(['tag_name', 'suggested_action']).count()
    print(action_dist)
    
    # Analyze element types
    print("\n=== Element Types ===")
    element_types = train_df['tag_name'].value_counts()
    print(element_types)
    
    # Look at some example elements
    print("\n=== Sample Elements ===")
    samples = train_df[['tag_name', 'inner_text', 'suggested_action']].head(5)
    print(samples)

if __name__ == "__main__":
    main()
