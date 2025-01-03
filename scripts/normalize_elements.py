"""
Normalize custom elements to standard HTML elements and extract generic features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
from typing import Dict, Any

class ElementNormalizer:
    def __init__(self):
        self.custom_element_map = {
            # YouTube specific mappings
            'yt-formatted-string': 'span',
            'yt-icon-button': 'button',
            'yt-attributed-string': 'span',
            'yt-icon': 'img',
            'yt-interaction': 'div',
            'tp-yt-paper-button': 'button',
            
            # Common custom elements
            'app-root': 'div',
            'app-header': 'header',
            'app-footer': 'footer',
            'custom-button': 'button',
            'custom-input': 'input',
            'custom-select': 'select',
            
            # Framework-specific elements
            'mat-button': 'button',
            'mat-input': 'input',
            'mat-select': 'select',
            'mat-icon': 'img',
            'mat-card': 'div',
            
            # Add more mappings as needed
        }
        
        self.semantic_roles = {
            # Navigation elements
            'nav': 'navigation',
            'a': 'navigation',
            'menu': 'navigation',
            'header': 'navigation',
            'footer': 'navigation',
            
            # Interactive elements
            'button': 'interactive',
            'input': 'interactive',
            'select': 'interactive',
            'textarea': 'interactive',
            'label': 'interactive',
            'form': 'interactive',
            
            # Content elements
            'p': 'content',
            'h1': 'content',
            'h2': 'content',
            'h3': 'content',
            'h4': 'content',
            'h5': 'content',
            'h6': 'content',
            'article': 'content',
            'section': 'content',
            'blockquote': 'content',
            'pre': 'content',
            'code': 'content',
            'dl': 'content',
            'dt': 'content',
            'dd': 'content',
            'li': 'content',
            
            # Media elements
            'img': 'media',
            'video': 'media',
            'audio': 'media',
            'picture': 'media',
            'source': 'media',
            'figure': 'media',
            'figcaption': 'media',
            'svg': 'media',
            'canvas': 'media',
            
            # Container elements
            'div': 'container',
            'span': 'container',
            'main': 'container',
            'aside': 'container'
        }

    def safe_str(self, value: Any) -> str:
        """Safely convert value to string, handling NaN and None."""
        if pd.isna(value) or value is None:
            return ''
        return str(value)

    def extract_semantic_features(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic features from an element."""
        features = {
            'semantic_role': 'unknown',
            'is_interactive': False,
            'is_visible': bool(row.get('is_visible', False)),
            'has_text': bool(self.safe_str(row.get('inner_text', '')).strip()),
            'normalized_tag': self.normalize_tag(self.safe_str(row.get('tag_name', ''))),
            'element_type': 'unknown'
        }
        
        # Get attributes
        attrs = row.get('attributes', {})
        if isinstance(attrs, str):
            try:
                attrs = ast.literal_eval(attrs)
            except:
                attrs = {}
        
        # Determine semantic role
        normalized_tag = features['normalized_tag']
        features['semantic_role'] = self.semantic_roles.get(normalized_tag, 'unknown')
        
        # Check for ARIA roles
        aria_role = attrs.get('role', '').lower()
        if aria_role:
            if aria_role in ['button', 'link', 'menuitem', 'tab', 'checkbox', 'radio']:
                features['semantic_role'] = 'interactive'
            elif aria_role in ['article', 'document', 'feed', 'main', 'region']:
                features['semantic_role'] = 'content'
            elif aria_role in ['navigation', 'menu', 'menubar', 'toolbar']:
                features['semantic_role'] = 'navigation'
            elif aria_role in ['img', 'figure']:
                features['semantic_role'] = 'media'
        
        # Determine if interactive
        features['is_interactive'] = (
            normalized_tag in ['button', 'a', 'input', 'select', 'textarea', 'label'] or
            (aria_role in ['button', 'link', 'menuitem', 'tab', 'checkbox', 'radio']) or
            bool(attrs.get('onclick')) or
            bool(attrs.get('href')) or
            'button' in str(attrs.get('class', '')).lower() or
            any(attr.startswith('on') for attr in attrs.keys())  # Any event handler
        )
        
        # Determine element type based on attributes and context
        if features['is_interactive']:
            features['element_type'] = 'interactive'
        elif features['semantic_role'] == 'navigation':
            features['element_type'] = 'navigation'
        elif features['semantic_role'] == 'content':
            features['element_type'] = 'content'
        elif features['semantic_role'] == 'media':
            features['element_type'] = 'media'
        else:
            features['element_type'] = 'container'
            
        return features

    def normalize_tag(self, tag_name: str) -> str:
        """Normalize custom element tags to standard HTML tags."""
        if not tag_name or pd.isna(tag_name):
            return 'div'  # Default to div for unknown elements
            
        tag_name = str(tag_name).lower()
        return self.custom_element_map.get(tag_name, tag_name)

    def process_dataset(self, input_file: Path, output_file: Path):
        """Process a dataset and normalize its elements."""
        try:
            # Read the dataset
            df = pd.read_csv(input_file)
            print(f"Processing {len(df)} elements from {input_file}")
            
            # Extract features for each element
            normalized_data = []
            for _, row in df.iterrows():
                # Convert string representations of dicts to actual dicts
                if isinstance(row.get('attributes'), str):
                    try:
                        attrs = ast.literal_eval(row['attributes'])
                        row['attributes'] = attrs if isinstance(attrs, dict) else {}
                    except:
                        row['attributes'] = {}
                
                # Extract semantic features
                features = self.extract_semantic_features(row)
                
                # Combine original data with new features
                element_data = {
                    'url': self.safe_str(row.get('url', '')),
                    'original_tag': self.safe_str(row.get('tag_name', '')),
                    'normalized_tag': features['normalized_tag'],
                    'semantic_role': features['semantic_role'],
                    'element_type': features['element_type'],
                    'is_interactive': features['is_interactive'],
                    'is_visible': features['is_visible'],
                    'has_text': features['has_text'],
                    'inner_text': self.safe_str(row.get('inner_text', '')),
                    'selector': self.safe_str(row.get('selector', '')),
                    'attributes': row.get('attributes', {})
                }
                
                normalized_data.append(element_data)
            
            # Create normalized DataFrame
            normalized_df = pd.DataFrame(normalized_data)
            
            # Save normalized data
            normalized_df.to_csv(output_file, index=False)
            print(f"Saved normalized data to {output_file}")
            
            # Print statistics
            print("\nElement type distribution:")
            print(normalized_df['element_type'].value_counts())
            print("\nSemantic role distribution:")
            print(normalized_df['semantic_role'].value_counts())
            print("\nOriginal vs Normalized tags (sample):")
            tag_comparison = pd.DataFrame({
                'original': normalized_df['original_tag'],
                'normalized': normalized_df['normalized_tag']
            }).drop_duplicates()
            print(tag_comparison[tag_comparison['original'] != tag_comparison['normalized']].head(10))
            
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            raise  # Re-raise the exception for debugging

def main():
    normalizer = ElementNormalizer()
    
    # Create output directories
    output_dir = Path('training_data/normalized')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process YouTube data
    youtube_input = Path('training_data/youtube/raw_data.csv')
    if youtube_input.exists():
        youtube_output = output_dir / 'youtube_normalized.csv'
        normalizer.process_dataset(youtube_input, youtube_output)
    
    # Process Automation Exercise data
    auto_input = Path('training_data/automation_exercise/processed/train.csv')
    if auto_input.exists():
        auto_output = output_dir / 'automation_exercise_normalized.csv'
        normalizer.process_dataset(auto_input, auto_output)
    
    # Process diverse site data
    diverse_sites_dir = Path('training_data/diverse_sites')
    if diverse_sites_dir.exists():
        print("\nProcessing diverse site data:")
        for file in diverse_sites_dir.glob('*_raw.csv'):
            try:
                output_file = output_dir / f"{file.stem.replace('_raw', '')}_normalized.csv"
                normalizer.process_dataset(file, output_file)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main()
