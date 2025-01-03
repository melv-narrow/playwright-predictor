"""
Compare patterns across different websites' training data to identify
common UI patterns and site-specific variations.
"""

import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import ast

class WebsitePatternAnalyzer:
    def __init__(self):
        self.sites_data = {}
        self.common_patterns = {}
        
    def load_site_data(self, site_name: str, data_path: Path):
        """Load and preprocess data for a specific site."""
        try:
            df = pd.read_csv(data_path)
            
            # Safely evaluate string representations of dicts
            if 'attributes' in df.columns:
                df['attributes'] = df['attributes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
            if 'bounding_box' in df.columns:
                df['bounding_box'] = df['bounding_box'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
                
            self.sites_data[site_name] = df
            print(f"Loaded {len(df)} elements from {site_name}")
            print(f"Columns available: {', '.join(df.columns)}")
        except Exception as e:
            print(f"Error loading {site_name} data: {str(e)}")

    def analyze_common_elements(self):
        """Analyze common UI elements across sites."""
        common_tags = {}
        common_roles = {}
        common_patterns = {}
        
        for site, df in self.sites_data.items():
            # Analyze tag distribution
            if 'tag_name' in df.columns:
                tag_dist = df['tag_name'].value_counts()
                common_tags[site] = tag_dist
            
            # Extract UI patterns
            patterns = self.extract_ui_patterns(df)
            common_patterns[site] = patterns
        
        return {
            'tags': common_tags,
            'patterns': common_patterns
        }
    
    def extract_ui_patterns(self, df: pd.DataFrame) -> Dict:
        """Extract common UI patterns from elements."""
        patterns = {
            'navigation': 0,
            'content': 0,
            'interaction': 0,
            'media': 0,
            'form': 0
        }
        
        for _, row in df.iterrows():
            tag_name = row.get('tag_name', '')
            
            # Skip if no tag name
            if not isinstance(tag_name, str):
                continue
                
            # Convert tag_name to lowercase for comparison
            tag_name = tag_name.lower()
                
            # Check for navigation elements
            if tag_name == 'nav' or 'nav' in tag_name:
                patterns['navigation'] += 1
            
            # Check for content elements
            elif tag_name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article']:
                patterns['content'] += 1
            
            # Check for interaction elements
            elif tag_name in ['button', 'a']:
                patterns['interaction'] += 1
            
            # Check for media elements
            elif tag_name in ['video', 'img', 'audio']:
                patterns['media'] += 1
            
            # Check for form elements
            elif tag_name in ['form', 'input', 'select', 'textarea']:
                patterns['form'] += 1
        
        return patterns

    def plot_comparisons(self):
        """Generate comparative visualizations."""
        analysis = self.analyze_common_elements()
        
        # Plot tag distributions
        plt.figure(figsize=(15, 8))
        
        if analysis['tags']:
            # Get common tags across all sites
            all_tags = set()
            for tags in analysis['tags'].values():
                all_tags.update(tags.index)
            
            # Create a DataFrame with all tags
            tag_data = {}
            for site, tags in analysis['tags'].items():
                tag_data[site] = [tags.get(tag, 0) for tag in all_tags]
            
            tag_df = pd.DataFrame(tag_data, index=list(all_tags))
            tag_df.plot(kind='bar')
            plt.title('Element Tag Distribution Across Sites')
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('training_data/analysis/tag_distribution.png', bbox_inches='tight')
        plt.close()
        
        # Plot UI patterns
        plt.figure(figsize=(12, 6))
        if analysis['patterns']:
            patterns_df = pd.DataFrame.from_dict(analysis['patterns'], orient='index')
            patterns_df.plot(kind='bar', width=0.8)
            plt.title('UI Pattern Distribution Across Sites')
            plt.xlabel('Website')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('training_data/analysis/pattern_distribution.png', bbox_inches='tight')
        plt.close()

def main():
    # Create analysis directory
    analysis_dir = Path('training_data/analysis')
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = WebsitePatternAnalyzer()
    
    # Load data from different sites
    analyzer.load_site_data('youtube', Path('training_data/youtube/raw_data.csv'))
    analyzer.load_site_data('automation_exercise', Path('training_data/automation_exercise/processed/train.csv'))
    
    # Generate analysis
    analyzer.plot_comparisons()
    
    # Print summary statistics
    for site, df in analyzer.sites_data.items():
        print(f"\n=== {site.upper()} Statistics ===")
        print(f"Total elements: {len(df)}")
        if 'tag_name' in df.columns:
            print("\nTop 10 element types:")
            print(df['tag_name'].value_counts().head(10))
        
        # Print available attributes
        print("\nAvailable attributes:")
        print(df.columns.tolist())

if __name__ == "__main__":
    main()
