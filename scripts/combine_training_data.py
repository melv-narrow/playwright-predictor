"""
Combine website element data with Playwright test patterns for training.
This script merges the normalized website data with Playwright documentation examples
to create a comprehensive training dataset.
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CombinedDataPreparator:
    def __init__(self):
        self.normalized_dir = Path('training_data/normalized')
        self.playwright_data_dir = Path('training_data/playwright_docs/training')
        self.combined_dir = Path('training_data/combined')
        self.combined_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    def load_element_data(self) -> pd.DataFrame:
        """Load the normalized website element data."""
        dfs = []
        for file in self.normalized_dir.glob('*_normalized.csv'):
            try:
                df = pd.read_csv(file)
                df['source'] = file.stem.replace('_normalized', '')
                dfs.append(df)
                logger.info(f"Loaded {len(df)} samples from {file.name}")
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
        
        if not dfs:
            raise ValueError("No normalized data files found")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Split into train/val/test
        train_df, temp_df = train_test_split(combined_df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.67, random_state=42)
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        logger.info(f"Total samples: {len(final_df)} (train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)})")
        
        return final_df

    def load_playwright_data(self) -> Dict[str, List[Dict]]:
        """Load the processed Playwright test data."""
        playwright_data = {}
        for split in ['train', 'val', 'test']:
            samples = []
            try:
                with open(self.playwright_data_dir / f'{split}.jsonl') as f:
                    for line in f:
                        samples.append(json.loads(line))
                playwright_data[split] = samples
                logger.info(f"Loaded {len(samples)} {split} Playwright samples")
            except Exception as e:
                logger.error(f"Error loading {split}.jsonl: {str(e)}")
        
        return playwright_data

    def create_combined_samples(self, 
                              element_data: pd.DataFrame, 
                              playwright_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Create combined training samples."""
        combined_data = {'train': [], 'val': [], 'test': []}
        
        # Process element data
        for _, row in element_data.iterrows():
            split = row['split']
            
            # Create element-based sample
            sample = {
                'instruction': 'Analyze this website element and suggest appropriate test cases:',
                'input': f"""
Element Type: {row['element_type']}
Semantic Role: {row['semantic_role']}
Text Content: {row['inner_text']}
Selector: {row['selector']}
Attributes: {row['attributes']}
Interactive: {row['is_interactive']}
Visible: {row['is_visible']}
""".strip(),
                'metadata': {
                    'type': 'element_analysis',
                    'element_type': row['element_type'],
                    'semantic_role': row['semantic_role']
                }
            }
            
            # Find matching Playwright examples
            matching_examples = []
            for example in playwright_data[split]:
                if self._is_relevant_example(example, row):
                    matching_examples.append(example['output'])
            
            if matching_examples:
                sample['output'] = '\n\n'.join(matching_examples[:2])  # Use top 2 matching examples
            else:
                # Generate a basic test template
                sample['output'] = self._generate_test_template(row)
            
            combined_data[split].append(sample)
        
        # Add Playwright examples
        for split, samples in playwright_data.items():
            combined_data[split].extend(samples)
        
        return combined_data

    def _is_relevant_example(self, example: Dict[str, Any], element: pd.Series) -> bool:
        """Check if a Playwright example is relevant for an element."""
        code = example['output'].lower()
        element_type = element['element_type'].lower()
        semantic_role = element['semantic_role'].lower()
        
        # Check for relevant patterns
        if element['is_interactive']:
            return any(pattern in code for pattern in [
                'click(', 'type(', 'fill(', 'select(', 'check(', 'uncheck('
            ])
        
        if semantic_role in ['heading', 'title']:
            return 'expect' in code and any(pattern in code for pattern in [
                'tohavetitle', 'tohavetext'
            ])
        
        if element_type in ['input', 'textarea']:
            return any(pattern in code for pattern in [
                'fill(', 'type(', 'getbyplaceholder'
            ])
        
        return False

    def _generate_test_template(self, element: pd.Series) -> str:
        """Generate a basic test template for an element."""
        selector = element['selector']
        
        if element['is_interactive']:
            return f"""
test('should interact with {element['semantic_role']}', async ({ page }) => {{
    const element = page.locator('{selector}');
    await expect(element).toBeVisible();
    await element.click();
}});""".strip()
        
        return f"""
test('should verify {element['semantic_role']}', async ({ page }) => {{
    const element = page.locator('{selector}');
    await expect(element).toBeVisible();
    await expect(element).toHaveText('{element['inner_text']}');
}});""".strip()

    def save_combined_data(self, data: Dict[str, List[Dict]]) -> None:
        """Save the combined dataset."""
        for split, samples in data.items():
            filepath = self.combined_dir / f'{split}.jsonl'
            with open(filepath, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            logger.info(f"Saved {len(samples)} combined {split} samples")
        
        # Save metadata
        metadata = {
            'total_samples': sum(len(samples) for samples in data.values()),
            'split_sizes': {split: len(samples) for split, samples in data.items()},
            'instruction_types': list(set(
                sample['metadata'].get('type', 'unknown') 
                for samples in data.values() 
                for sample in samples
            ))
        }
        
        with open(self.combined_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def process(self):
        """Main processing method."""
        logger.info("Loading element data...")
        element_data = self.load_element_data()
        
        logger.info("Loading Playwright data...")
        playwright_data = self.load_playwright_data()
        
        logger.info("Creating combined samples...")
        combined_data = self.create_combined_samples(element_data, playwright_data)
        
        logger.info("Saving combined data...")
        self.save_combined_data(combined_data)
        
        logger.info("Done!")

def main():
    preparator = CombinedDataPreparator()
    preparator.process()

if __name__ == "__main__":
    main()
