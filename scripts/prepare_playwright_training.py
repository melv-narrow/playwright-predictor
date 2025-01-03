"""
Prepare training data from Playwright documentation examples.
This script converts extracted test examples into a format suitable for fine-tuning
the model to generate Playwright tests.
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlaywrightTrainingPreparator:
    def __init__(self):
        self.data_dir = Path('training_data/playwright_docs')
        self.examples_dir = self.data_dir / 'examples'
        self.processed_dir = self.data_dir / 'training'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_examples(self) -> List[Dict]:
        """Load all test examples from JSON files."""
        examples = []
        for file in self.examples_dir.glob('example_*.json'):
            with open(file) as f:
                example = json.load(f)
                examples.append(example)
        return examples

    def create_training_samples(self, examples: List[Dict]) -> List[Dict]:
        """Convert examples into training samples."""
        samples = []
        
        for example in examples:
            # Extract key information
            code = example['code']
            context = example['context']
            pattern = example['pattern']
            url = example['source_url']
            
            # Create instruction for different scenarios
            instructions = [
                # Basic test generation
                {
                    'instruction': f'Generate a Playwright test for the following scenario: {context}',
                    'input': '',
                    'output': code,
                    'metadata': {
                        'pattern': pattern,
                        'source': url
                    }
                },
                
                # Pattern-specific generation
                {
                    'instruction': f'Create a Playwright test that demonstrates the {pattern} pattern.',
                    'input': context,
                    'output': code,
                    'metadata': {
                        'pattern': pattern,
                        'source': url
                    }
                },
                
                # Test modification
                {
                    'instruction': 'Modify this test to follow Playwright best practices:',
                    'input': code.replace('expect', 'assert').replace('page.goto', 'page.navigate'),  # Create slightly modified version
                    'output': code,
                    'metadata': {
                        'pattern': pattern,
                        'source': url,
                        'type': 'modification'
                    }
                }
            ]
            
            samples.extend(instructions)
            
            # Add pattern-specific samples
            if pattern == 'assertion':
                samples.append({
                    'instruction': 'Add appropriate assertions to verify the page state:',
                    'input': '\n'.join([
                        'test("check homepage", async ({ page }) => {',
                        '  await page.goto("https://example.com");',
                        '  // Add assertions here',
                        '});'
                    ]),
                    'output': code,
                    'metadata': {
                        'pattern': pattern,
                        'source': url,
                        'type': 'completion'
                    }
                })
            elif pattern == 'navigation':
                samples.append({
                    'instruction': 'Create a test that navigates through multiple pages:',
                    'input': context,
                    'output': code,
                    'metadata': {
                        'pattern': pattern,
                        'source': url,
                        'type': 'navigation'
                    }
                })
            elif pattern == 'fixture':
                samples.append({
                    'instruction': 'Create a test fixture for setting up test data:',
                    'input': context,
                    'output': code,
                    'metadata': {
                        'pattern': pattern,
                        'source': url,
                        'type': 'fixture'
                    }
                })
            
            # Add website-specific test generation sample
            samples.append({
                'instruction': 'Generate a Playwright test suite for a website. The test should:',
                'input': '\n'.join([
                    '1. Navigate to the homepage',
                    '2. Verify key elements are present',
                    '3. Test main user interactions',
                    '4. Handle any errors or loading states',
                    '5. Follow Playwright best practices'
                ]),
                'output': code,
                'metadata': {
                    'pattern': pattern,
                    'source': url,
                    'type': 'website_test'
                }
            })
        
        return samples

    def save_training_data(self, samples: List[Dict], split: bool = True) -> None:
        """Save processed training data."""
        if split:
            # Split into train/val/test
            train_data, test_data = train_test_split(samples, test_size=0.2, random_state=42)
            train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
            
            # Save splits
            self._save_jsonl(train_data, self.processed_dir / 'train.jsonl')
            self._save_jsonl(val_data, self.processed_dir / 'val.jsonl')
            self._save_jsonl(test_data, self.processed_dir / 'test.jsonl')
            
            # Save metadata
            metadata = {
                'total_samples': len(samples),
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data),
                'patterns': list(set(s['metadata']['pattern'] for s in samples)),
                'types': list(set(s['metadata'].get('type', 'basic') for s in samples))
            }
            
            with open(self.processed_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test samples")
        else:
            self._save_jsonl(samples, self.processed_dir / 'all_data.jsonl')
            logger.info(f"Saved {len(samples)} samples")

    def _save_jsonl(self, data: List[Dict], filepath: Path) -> None:
        """Save data in JSONL format."""
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    def process(self) -> None:
        """Process all examples into training data."""
        logger.info("Loading examples...")
        examples = self.load_examples()
        logger.info(f"Loaded {len(examples)} examples")
        
        logger.info("Creating training samples...")
        samples = self.create_training_samples(examples)
        logger.info(f"Created {len(samples)} training samples")
        
        logger.info("Saving training data...")
        self.save_training_data(samples)
        logger.info("Done!")

def main():
    preparator = PlaywrightTrainingPreparator()
    preparator.process()

if __name__ == "__main__":
    main()
