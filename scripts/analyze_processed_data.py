"""
Analyze the processed training data to understand feature distributions
and relationships between elements and actions. Extracts semantic features
and generic patterns for better website adaptability.
"""

import pandas as pd
import json
from pathlib import Path
import re
from typing import Dict, List

def extract_semantic_features(row: pd.Series) -> Dict:
    """Extract semantic features from element attributes and context."""
    features = {
        'semantic_role': 'generic',
        'interaction_type': 'none',
        'ui_pattern': 'content',
        'hierarchy_level': 0
    }
    
    # Determine semantic role
    if 'role' in row['attributes']:
        features['semantic_role'] = row['attributes']['role']
    elif row['tag_name'] in ['nav', 'header', 'footer', 'main', 'aside']:
        features['semantic_role'] = row['tag_name']
    elif row['tag_name'] == 'a':
        features['semantic_role'] = 'navigation'
    elif row['tag_name'] in ['button', 'input', 'select', 'textarea']:
        features['semantic_role'] = 'interactive'
        
    # Determine interaction type
    if row['tag_name'] in ['button', 'a'] or row['suggested_action'] == 'click':
        features['interaction_type'] = 'clickable'
    elif row['tag_name'] in ['input', 'textarea']:
        features['interaction_type'] = 'input'
    elif row['tag_name'] in ['select']:
        features['interaction_type'] = 'select'
        
    # Identify UI patterns
    if 'form' in row['selector'].lower():
        features['ui_pattern'] = 'form'
    elif 'nav' in row['selector'].lower():
        features['ui_pattern'] = 'navigation'
    elif 'search' in row['selector'].lower():
        features['ui_pattern'] = 'search'
    elif 'list' in row['selector'].lower():
        features['ui_pattern'] = 'list'
        
    # Determine visual hierarchy
    if any(x in str(row['attributes']).lower() for x in ['primary', 'main']):
        features['hierarchy_level'] = 1
    elif any(x in str(row['attributes']).lower() for x in ['secondary']):
        features['hierarchy_level'] = 2
        
    return features

def categorize_element(row: pd.Series) -> str:
    """Categorize element into generic types."""
    if row['tag_name'] in ['nav', 'a'] or 'nav' in row['selector'].lower():
        return 'navigation'
    elif row['tag_name'] in ['button', 'input', 'select', 'textarea']:
        return 'interactive'
    elif row['tag_name'] in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span']:
        return 'content'
    elif row['tag_name'] in ['div', 'section', 'article']:
        return 'container'
    else:
        return 'other'

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
    
    # Extract semantic features
    semantic_features = train_df.apply(extract_semantic_features, axis=1)
    train_df = pd.concat([train_df, pd.DataFrame(semantic_features.tolist())], axis=1)
    
    # Categorize elements
    train_df['element_category'] = train_df.apply(categorize_element, axis=1)
    
    # Print analysis results
    print("\n=== Element Categories ===")
    print(train_df['element_category'].value_counts())
    
    print("\n=== Semantic Roles ===")
    print(train_df['semantic_role'].value_counts())
    
    print("\n=== UI Patterns ===")
    print(train_df['ui_pattern'].value_counts())
    
    print("\n=== Interaction Types ===")
    print(train_df['interaction_type'].value_counts())
    
    # Analyze relationships
    print("\n=== Category vs Action Correlation ===")
    category_action = pd.crosstab(train_df['element_category'], train_df['suggested_action'])
    print(category_action)

if __name__ == "__main__":
    main()
