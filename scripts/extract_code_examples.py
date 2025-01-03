"""
Script to extract code examples from downloaded Playwright documentation markdown files.
"""

import os
import json
import logging
import re
from typing import Dict, List
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeExampleExtractor:
    def __init__(self):
        self.markdown_dir = os.path.join("training_data", "playwright_docs", "markdown")
        self.output_dir = os.path.join("training_data", "playwright_docs")
        self.examples_dir = os.path.join(self.output_dir, "examples")
        self.patterns_dir = os.path.join(self.output_dir, "patterns")
        
        # Create output directories
        os.makedirs(self.examples_dir, exist_ok=True)
        os.makedirs(self.patterns_dir, exist_ok=True)

    def is_test_related(self, code: str) -> bool:
        """Check if code block is related to testing."""
        # Common test file patterns
        test_file_patterns = [
            r'\.spec\.[jt]s$',
            r'\.test\.[jt]s$',
            r'_test\.[jt]s$'
        ]
        
        # Test-related imports
        test_imports = [
            'import { test }',
            'import { expect }',
            'const { test }',
            'const { expect }',
            'import { chromium }',
            'import { firefox }',
            'import { webkit }',
            '@playwright/test'
        ]
        
        # Test declarations and assertions
        test_indicators = [
            # Test declarations
            'test(', 'test.describe(', 'test.beforeEach(', 'test.afterEach(',
            'describe(', 'it(', 'beforeAll(', 'afterAll(',
            
            # Playwright specific
            'page.goto', 'page.click', 'page.fill', 'page.type',
            'locator(', 'getByRole(', 'getByText(', 'getByLabel(',
            'expect(', 'toHaveText', 'toBeVisible', 'toBeEnabled',
            
            # Common test operations
            'await page', 'fixture', 'test.use', 'browser.newContext',
            'screenshot', 'waitFor', 'evaluate', 'assert',
            
            # Test configuration
            'playwright.config', 'testConfig', 'testMatch',
            'testDir', 'testIgnore', 'reporter'
        ]
        
        code_lower = code.lower()
        
        # Check for test file patterns
        if any(re.search(pattern, code_lower) for pattern in test_file_patterns):
            return True
            
        # Check for test-related imports
        if any(imp.lower() in code_lower for imp in test_imports):
            return True
            
        # Check for test indicators
        return any(indicator.lower() in code_lower for indicator in test_indicators)

    def extract_code_blocks(self, markdown: str, source_url: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        
        # Split into lines for processing
        lines = markdown.split('\n')
        current_block = []
        in_code_block = False
        language = None
        block_start_line = 0
        
        for i, line in enumerate(lines):
            # Check for triple backtick code blocks
            if line.startswith('```'):
                if not in_code_block:
                    # Start of code block
                    in_code_block = True
                    language = line[3:].strip()  # Get language after ```
                    block_start_line = i
                else:
                    # End of code block
                    in_code_block = False
                    if current_block and language in ['typescript', 'js', 'javascript']:
                        code_blocks.append(self.create_code_block(
                            '\n'.join(current_block),
                            language,
                            lines,
                            block_start_line,
                            source_url
                        ))
                    current_block = []
                continue
            
            # Check for 4-space indented code blocks
            if not in_code_block and line.startswith('    '):
                if not current_block:
                    # Start of indented block
                    block_start_line = i
                current_block.append(line[4:])  # Remove 4 spaces
                continue
            elif current_block and not line.startswith('    ') and line.strip():
                # End of indented block
                # Try to detect language from content
                code = '\n'.join(current_block)
                if any(indicator in code for indicator in ['import', 'const', 'let', 'function', '=>', 'await']):
                    language = 'javascript'
                    code_blocks.append(self.create_code_block(
                        code,
                        language,
                        lines,
                        block_start_line,
                        source_url
                    ))
                current_block = []
            
            # Add line to current block if we're in one
            if in_code_block:
                current_block.append(line)
        
        return code_blocks
    
    def create_code_block(self, code: str, language: str, lines: List[str], block_start: int, source_url: str) -> Dict[str, str]:
        """Create a code block dictionary with context."""
        # Look back up to 3 lines for context
        context_lines = []
        for i in range(max(0, block_start - 3), block_start):
            line = lines[i].strip()
            if line and not line.startswith('```'):
                # Remove markdown headers and list markers
                line = re.sub(r'^#+\s*', '', line)
                line = re.sub(r'^[-*+]\s+', '', line)
                context_lines.append(line)
        
        return {
            'language': language,
            'code': code.strip(),
            'context': ' | '.join(context_lines),
            'source_url': source_url
        }

    def categorize_test_pattern(self, code: str) -> str:
        """Categorize the test pattern in the code."""
        patterns = {
            'page_object': ['class.*Page', 'extends.*Page'],
            'fixture': ['fixture', 'beforeAll', 'afterAll'],
            'assertion': ['expect', 'assert', 'should'],
            'navigation': ['goto', 'click', 'navigate'],
            'form_interaction': ['fill', 'type', 'select'],
            'wait_strategy': ['waitFor', 'wait'],
            'error_handling': ['try.*catch', 'expect.*toThrow']
        }
        
        for pattern, indicators in patterns.items():
            if any(re.search(ind, code, re.IGNORECASE) for ind in indicators):
                return pattern
        
        return 'general'

    def process_markdown_file(self, filepath: str) -> List[Dict[str, str]]:
        """Process a single markdown file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract source URL from first line
            source_url = content.split('\n')[0].replace('Source: ', '').strip()
            
            # Extract code blocks
            code_blocks = self.extract_code_blocks(content, source_url)
            
            # Filter test-related blocks
            test_blocks = []
            for block in code_blocks:
                if self.is_test_related(block['code']):
                    test_blocks.append(block)
            
            if test_blocks:
                logger.info(f"Found {len(test_blocks)} test examples in {os.path.basename(filepath)}")
            else:
                logger.debug(f"No test examples found in {os.path.basename(filepath)}")
            
            return test_blocks
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return []

    def extract_all_examples(self):
        """Extract code examples from all markdown files."""
        logger.info("Starting code example extraction...")
        
        all_examples = []
        
        # Process each markdown file
        for filename in os.listdir(self.markdown_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(self.markdown_dir, filename)
                logger.info(f"Processing {filename}")
                examples = self.process_markdown_file(filepath)
                all_examples.extend(examples)
        
        if not all_examples:
            logger.warning("No test examples were collected!")
            return
        
        # Categorize and save examples
        logger.info(f"Processing {len(all_examples)} total examples")
        for i, example in enumerate(all_examples):
            pattern = self.categorize_test_pattern(example['code'])
            example['pattern'] = pattern
            
            # Save individual example
            filename = f"example_{i}_{pattern}.json"
            with open(os.path.join(self.examples_dir, filename), 'w') as f:
                json.dump(example, f, indent=2)
        
        # Create summary DataFrame
        df = pd.DataFrame(all_examples)
        df.to_csv(os.path.join(self.output_dir, 'test_examples.csv'), index=False)
        
        # Generate pattern statistics
        pattern_stats = df['pattern'].value_counts()
        pattern_stats.to_csv(os.path.join(self.patterns_dir, 'pattern_distribution.csv'))
        logger.info("Pattern distribution:\n" + str(pattern_stats))
        
        logger.info(f"Collected {len(all_examples)} test examples")

def main():
    extractor = CodeExampleExtractor()
    extractor.extract_all_examples()

if __name__ == "__main__":
    main()
