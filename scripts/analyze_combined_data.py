"""
Analyze combined normalized data to identify patterns and gaps in our training data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import numpy as np

class DatasetAnalyzer:
    def __init__(self):
        self.normalized_dir = Path('training_data/normalized')
        self.analysis_dir = Path('training_data/analysis')
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all normalized datasets."""
        datasets = {}
        
        # Only load normalized datasets
        for file in self.normalized_dir.glob('*_normalized.csv'):
            try:
                df = pd.read_csv(file)
                source_name = file.stem.replace('_normalized', '')
                df['source'] = source_name
                datasets[source_name] = df
                print(f"Loaded {len(df)} records from {file.name}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                
        return datasets
    
    def analyze_element_coverage(self, datasets: Dict[str, pd.DataFrame]):
        """Analyze element type coverage across datasets."""
        combined_df = pd.concat(datasets.values(), ignore_index=True)
        
        # Element type distribution
        plt.figure(figsize=(15, 8))
        element_dist = combined_df.groupby(['source', 'element_type']).size().unstack(fill_value=0)
        ax = element_dist.plot(kind='bar', stacked=True)
        plt.title('Element Type Distribution by Source')
        plt.xlabel('Source')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Element Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels on bars
        for c in ax.containers:
            ax.bar_label(c, label_type='center')
            
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'element_distribution.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Semantic role coverage with percentages
        plt.figure(figsize=(15, 8))
        role_dist = combined_df.groupby(['source', 'semantic_role']).size().unstack(fill_value=0)
        role_percentages = role_dist.div(role_dist.sum(axis=1), axis=0) * 100
        ax = role_percentages.plot(kind='bar', stacked=True)
        plt.title('Semantic Role Distribution by Source (%)')
        plt.xlabel('Source')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Semantic Role', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add percentage labels
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f%%', label_type='center')
            
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'semantic_distribution_percent.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Print detailed statistics
        print("\n=== Overall Statistics ===")
        print(f"Total elements: {len(combined_df)}")
        
        print("\nElement type distribution:")
        type_dist = combined_df['element_type'].value_counts()
        type_percent = type_dist / len(combined_df) * 100
        for t, count in type_dist.items():
            print(f"{t:12} {count:5d} ({type_percent[t]:5.1f}%)")
        
        print("\nSemantic role distribution:")
        role_dist = combined_df['semantic_role'].value_counts()
        role_percent = role_dist / len(combined_df) * 100
        for r, count in role_dist.items():
            print(f"{r:12} {count:5d} ({role_percent[r]:5.1f}%)")
        
        # Analyze interactive elements
        interactive_df = combined_df[combined_df['element_type'] == 'interactive']
        print("\n=== Interactive Elements Analysis ===")
        print("Interactive elements by source:")
        interactive_by_source = interactive_df.groupby('source').size()
        interactive_percent = interactive_by_source / combined_df.groupby('source').size() * 100
        for source in interactive_by_source.index:
            print(f"{source:20} {interactive_by_source[source]:5d} ({interactive_percent[source]:5.1f}%)")
        
        # Analyze visibility patterns
        print("\n=== Visibility Analysis ===")
        visibility_dist = pd.crosstab(
            combined_df['source'], 
            combined_df['is_visible'], 
            normalize='index'
        ) * 100
        print("Visibility percentages by source:")
        print(visibility_dist.round(1))
        
        # Save detailed analysis
        analysis = {
            'total_elements': int(len(combined_df)),
            'elements_by_source': {k: int(v) for k, v in combined_df.groupby('source').size().to_dict().items()},
            'element_type_distribution': {k: int(v) for k, v in combined_df['element_type'].value_counts().to_dict().items()},
            'semantic_role_distribution': {k: int(v) for k, v in combined_df['semantic_role'].value_counts().to_dict().items()},
            'interactive_elements': {
                'total': int(len(interactive_df)),
                'by_source': {k: int(v) for k, v in interactive_df.groupby('source').size().to_dict().items()}
            },
            'visibility': {
                'visible': int(combined_df['is_visible'].sum()),
                'total': int(len(combined_df))
            }
        }
        
        with open(self.analysis_dir / 'detailed_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis

    def identify_gaps(self, analysis: Dict):
        """Identify gaps in our training data."""
        gaps = []
        
        # Check element type balance
        element_dist = pd.Series(analysis['element_type_distribution'])
        total = element_dist.sum()
        
        print("\n=== Element Type Balance ===")
        for element_type, count in element_dist.items():
            percentage = (count / total) * 100
            print(f"{element_type:12} {percentage:5.1f}%")
            if percentage < 10:  # Less than 10% representation
                gaps.append(f"Under-represented element type: {element_type} ({percentage:.1f}%)")
        
        # Check semantic role coverage
        role_dist = pd.Series(analysis['semantic_role_distribution'])
        total_roles = role_dist.sum()
        
        print("\n=== Semantic Role Coverage ===")
        for role, count in role_dist.items():
            percentage = (count / total_roles) * 100
            print(f"{role:12} {percentage:5.1f}%")
            if percentage < 5:  # Less than 5% representation
                gaps.append(f"Under-represented semantic role: {role} ({percentage:.1f}%)")
        
        # Check source balance
        source_dist = pd.Series(analysis['elements_by_source'])
        source_percentage = source_dist / total * 100
        
        print("\n=== Source Distribution ===")
        for source, count in source_dist.items():
            percentage = (count / total) * 100
            print(f"{source:20} {percentage:5.1f}%")
            if percentage < 5:  # Less than 5% representation
                gaps.append(f"Under-represented source: {source} ({percentage:.1f}%)")
        
        print("\n=== Identified Gaps ===")
        for gap in gaps:
            print(f"- {gap}")
        
        return gaps

def main():
    analyzer = DatasetAnalyzer()
    datasets = analyzer.load_all_datasets()
    analysis = analyzer.analyze_element_coverage(datasets)
    gaps = analyzer.identify_gaps(analysis)
    
    # Suggest improvements
    print("\n=== Suggested Improvements ===")
    if gaps:
        print("Based on the identified gaps, consider:")
        for gap in gaps:
            if "element type" in gap:
                print(f"- Collect more {gap.split(':')[1].split('(')[0].strip()} elements")
            elif "semantic role" in gap:
                print(f"- Add sites with more {gap.split(':')[1].split('(')[0].strip()} content")
            elif "source" in gap:
                print(f"- Expand data collection from {gap.split(':')[1].split('(')[0].strip()}")
    else:
        print("No significant gaps identified in the current dataset.")

if __name__ == "__main__":
    main()
