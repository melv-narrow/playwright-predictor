"""
Script to scrape and process Playwright documentation for test examples and patterns.
"""

import os
import json
import logging
from typing import Dict, List, Optional
import re
from urllib.parse import urljoin
import asyncio
from playwright.async_api import async_playwright, Page
import pandas as pd
import html2text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlaywrightDocScraper:
    def __init__(self):
        self.base_url = "https://playwright.dev/docs/intro"
        self.docs_base = "https://playwright.dev"
        self.output_dir = os.path.join("training_data", "playwright_docs")
        self.examples_dir = os.path.join(self.output_dir, "examples")
        self.patterns_dir = os.path.join(self.output_dir, "patterns")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.examples_dir, exist_ok=True)
        os.makedirs(self.patterns_dir, exist_ok=True)
        
        # Configure HTML to Markdown converter
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = True
        self.html2text.ignore_images = True
        self.html2text.ignore_emphasis = True
        self.html2text.body_width = 0  # Don't wrap lines

    async def is_test_related(self, code: str) -> bool:
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

    async def extract_code_blocks(self, markdown: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown content."""
        # Regular expression to match code blocks with language
        pattern = r'```(\w+)\n(.*?)```'
        matches = re.finditer(pattern, markdown, re.DOTALL)
        
        code_blocks = []
        
        # Split markdown into lines for context tracking
        lines = markdown.split('\n')
        current_line = 0
        
        for match in matches:
            language = match.group(1)
            code = match.group(2).strip()
            
            # Find the line number where this code block starts
            block_start = markdown.count('\n', 0, match.start())
            
            # Look back up to 3 lines for context
            context_lines = []
            for i in range(max(0, block_start - 3), block_start):
                if i < len(lines):
                    line = lines[i].strip()
                    if line and not line.startswith('```'):
                        # Remove markdown headers and list markers
                        line = re.sub(r'^#+\s*', '', line)
                        line = re.sub(r'^[-*+]\s+', '', line)
                        context_lines.append(line)
            
            if language in ['typescript', 'js', 'javascript']:
                code_blocks.append({
                    'language': language,
                    'code': code,
                    'context': ' | '.join(context_lines)
                })
        
        return code_blocks

    async def collect_doc_links(self, page: Page) -> List[str]:
        """Collect all documentation page links."""
        # Wait for navigation to load
        await page.wait_for_selector('nav a[href^="/docs/"]')
        
        # Get all documentation links
        links = await page.evaluate("""
            () => {
                const links = new Set();
                document.querySelectorAll('nav a[href^="/docs/"]').forEach(a => {
                    const href = a.getAttribute('href');
                    if (href) links.add(href);
                });
                return Array.from(links);
            }
        """)
        
        # Convert relative paths to absolute URLs and remove duplicates
        absolute_links = list(set(urljoin(self.docs_base, link) for link in links))
        logger.info(f"Found {len(absolute_links)} unique documentation pages")
        return absolute_links

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

    async def process_page(self, page: Page, url: str) -> List[Dict[str, str]]:
        """Process a single documentation page."""
        try:
            # Navigate and wait for content
            await page.goto(url, wait_until='networkidle')
            await page.wait_for_selector('article', state='attached')
            
            # Get the HTML content
            html_content = await page.evaluate('() => document.querySelector("article").outerHTML')
            
            # Convert HTML to Markdown
            markdown = self.html2text.handle(html_content)
            logger.debug(f"Converted markdown from {url}:")
            logger.debug(markdown[:500] + "..." if len(markdown) > 500 else markdown)
            
            # Extract code blocks
            code_blocks = await self.extract_code_blocks(markdown)
            logger.info(f"Found {len(code_blocks)} code blocks in {url}")
            
            # Filter test-related blocks
            test_blocks = []
            for block in code_blocks:
                if await self.is_test_related(block['code']):
                    block['source_url'] = url
                    test_blocks.append(block)
                    logger.debug(f"Found test block in {url}:")
                    logger.debug(block['code'][:200] + "..." if len(block['code']) > 200 else block['code'])
            
            if test_blocks:
                logger.info(f"Found {len(test_blocks)} test examples in {url}")
            else:
                logger.debug(f"No test examples found in {url}")
            
            return test_blocks
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return []

    async def scrape_documentation(self):
        """Main method to scrape Playwright documentation."""
        logger.info("Starting documentation scraping...")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # Collect all documentation links
            await page.goto(self.base_url)
            doc_links = await self.collect_doc_links(page)
            
            # Process each page
            all_examples = []
            for url in doc_links:
                logger.info(f"Processing {url}")
                examples = await self.process_page(page, url)
                all_examples.extend(examples)
            
            await browser.close()
            
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

async def main():
    scraper = PlaywrightDocScraper()
    await scraper.scrape_documentation()

if __name__ == "__main__":
    asyncio.run(main())
