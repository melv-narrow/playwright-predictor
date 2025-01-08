"""
Analyze websites and generate Playwright tests using the trained model.
"""

import logging
from pathlib import Path
import torch
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
import json
from typing import Dict, List
from finetune_playwright import EnhancedElementClassifier
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteAnalyzer:
    def __init__(
        self,
        model_path: str = 'models/playwright_enhanced_model.pt',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        self.model = EnhancedElementClassifier(
            base_model='microsoft/codebert-base',
            num_patterns=7,
            num_test_patterns=7
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Pattern mapping
        self.pattern_map = {
            0: 'assertion',
            1: 'navigation',
            2: 'fixture',
            3: 'wait_strategy',
            4: 'form_interaction',
            5: 'page_object',
            6: 'general'
        }
    
    def analyze_page(self, url: str) -> Dict:
        """Analyze a webpage and identify test patterns."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Navigate to URL
            logger.info(f"Analyzing page: {url}")
            page.goto(url)
            
            # Wait for key elements to load
            page.wait_for_selector('nav', state='visible')
            
            # Get page content
            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Analyze different sections
            forms = soup.find_all('form')
            links = soup.find_all('a')
            buttons = soup.find_all('button')
            inputs = soup.find_all('input')
            nav_items = soup.find_all('nav')
            headers = soup.find_all(['h1', 'h2', 'h3'])
            
            # Store elements by type
            elements = {
                'forms': forms,
                'links': links,
                'buttons': buttons,
                'inputs': inputs,
                'nav_items': nav_items,
                'headers': headers
            }
            
            # Generate test patterns for each element type
            test_patterns = self._analyze_elements(elements)
            
            browser.close()
            return test_patterns
    
    def _analyze_elements(self, elements: Dict) -> List[Dict]:
        """Analyze elements and determine test patterns."""
        test_patterns = []
        
        # Analyze forms with specific form types
        for form in elements['forms']:
            form_inputs = form.find_all('input')
            form_buttons = form.find_all('button')
            
            # Determine form type
            form_type = self._determine_form_type(form_inputs)
            instruction = f"Generate test for {form_type} form"
            
            pattern = self._predict_pattern(instruction, str(form))
            
            if pattern['confidence'] > 0.5:
                test_patterns.append({
                    'element_type': 'form',
                    'form_type': form_type,
                    'html': str(form),
                    'pattern': pattern['pattern'],
                    'confidence': pattern['confidence'],
                    'inputs': [{'id': inp.get('id'), 'type': inp.get('type'), 
                              'name': inp.get('name'), 'placeholder': inp.get('placeholder')} 
                             for inp in form_inputs],
                    'buttons': [{'id': btn.get('id'), 'text': btn.text.strip(), 
                               'type': btn.get('type')} for btn in form_buttons]
                })
        
        # Analyze standalone inputs (like search)
        for input_elem in elements['inputs']:
            if input_elem.parent.name != 'form':  # Only process inputs not in forms
                input_type = input_elem.get('type', 'text')
                placeholder = input_elem.get('placeholder', '').lower()
                
                if 'search' in placeholder:
                    instruction = "Generate test for search functionality"
                    pattern = self._predict_pattern(instruction, str(input_elem))
                    
                    if pattern['confidence'] > 0.5:
                        test_patterns.append({
                            'element_type': 'search',
                            'html': str(input_elem),
                            'pattern': pattern['pattern'],
                            'confidence': pattern['confidence'],
                            'input': {
                                'id': input_elem.get('id'),
                                'name': input_elem.get('name'),
                                'placeholder': placeholder
                            }
                        })
        
        # Analyze navigation menu with hierarchy
        nav_patterns = self._analyze_navigation(elements['nav_items'])
        test_patterns.extend(nav_patterns)
        
        # Analyze content and assertions
        content_patterns = self._analyze_content(elements)
        test_patterns.extend(content_patterns)
        
        return test_patterns
    
    def _determine_form_type(self, inputs) -> str:
        """Determine the type of form based on its inputs."""
        input_types = [inp.get('type', '').lower() for inp in inputs]
        placeholders = [inp.get('placeholder', '').lower() for inp in inputs]
        
        if 'password' in input_types:
            return 'authentication'
        elif 'email' in input_types and len(inputs) == 1:
            return 'subscription'
        elif any('search' in p for p in placeholders):
            return 'search'
        return 'general'
    
    def _analyze_navigation(self, nav_items) -> List[Dict]:
        """Analyze navigation elements with hierarchy."""
        patterns = []
        for nav in nav_items:
            nav_links = nav.find_all('a')
            parent_text = nav.find_parent(['div', 'section']).get('aria-label', '').strip()
            
            for link in nav_links:
                instruction = f"Generate test for navigation in {parent_text}"
                pattern = self._predict_pattern(instruction, str(link))
                
                if pattern['confidence'] > 0.5:
                    patterns.append({
                        'element_type': 'navigation',
                        'section': parent_text,
                        'html': str(link),
                        'pattern': pattern['pattern'],
                        'confidence': pattern['confidence'],
                        'href': link.get('href'),
                        'text': link.text.strip()
                    })
        return patterns
    
    def _analyze_content(self, elements) -> List[Dict]:
        """Analyze page content for assertions."""
        patterns = []
        
        # Analyze headers
        for header in elements['headers']:
            instruction = "Generate assertion test for header"
            pattern = self._predict_pattern(instruction, str(header))
            
            if pattern['confidence'] > 0.5:
                patterns.append({
                    'element_type': 'assertion',
                    'assertion_type': 'header',
                    'html': str(header),
                    'pattern': pattern['pattern'],
                    'confidence': pattern['confidence'],
                    'text': header.text.strip()
                })
        
        # Analyze key content sections
        sections = elements.get('sections', [])
        for section in sections:
            if section.get('class') and any('hero' in c for c in section['class']):
                instruction = "Generate assertion test for main content"
                pattern = self._predict_pattern(instruction, str(section))
                
                if pattern['confidence'] > 0.5:
                    patterns.append({
                        'element_type': 'assertion',
                        'assertion_type': 'content',
                        'html': str(section),
                        'pattern': pattern['pattern'],
                        'confidence': pattern['confidence'],
                        'text': section.text.strip()
                    })
        
        return patterns
    
    def _predict_pattern(self, instruction: str, input_text: str) -> Dict:
        """Predict test pattern for given input."""
        inputs = self.tokenizer(
            instruction + ' ' + input_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            pattern_probs = torch.softmax(outputs.pattern_logits, dim=-1)
            pattern_idx = torch.argmax(pattern_probs, dim=-1).item()
        
        return {
            'pattern': self.pattern_map[pattern_idx],
            'confidence': pattern_probs[0][pattern_idx].item()
        }
    
    def generate_tests(self, url: str, output_dir: str = 'generated_tests'):
        """Analyze website and generate Playwright tests."""
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Analyze page
        test_patterns = self.analyze_page(url)
        
        # Generate test file
        test_code = self._generate_test_code(url, test_patterns)
        
        # Save test file
        output_file = output_dir / f"{url.replace('https://', '').replace('/', '_')}_test.spec.ts"
        with open(output_file, 'w') as f:
            f.write(test_code)
        
        logger.info(f"Generated test saved to: {output_file}")
        return test_patterns
    
    def _generate_test_code(self, url: str, test_patterns: List[Dict]) -> str:
        """Generate Playwright test code from patterns."""
        test_code = [
            "import { test, expect, Page } from '@playwright/test';",
            "",
            "test.describe('GitHub Homepage Tests', () => {",
            "  let page: Page;",
            "",
            "  test.beforeEach(async ({ browser }) => {",
            "    page = await browser.newPage();",
            f"    await page.goto('{url}');",
            "    await page.waitForLoadState('networkidle');",
            "  });",
            "",
            "  test.afterEach(async () => {",
            "    await page?.close();",
            "  });",
            "",
            "  test('Basic Page Elements', async () => {",
            "    // Verify page load",
            "    await expect(page).toHaveURL(/.*github.com/);",
            "",
            "    // Verify key elements",
            "    await expect(page.getByRole('heading', { name: /Build.*software/i })).toBeVisible();",
            "    await expect(page.getByRole('button', { name: /Try.*Copilot/i })).toBeVisible();",
            "    await expect(page.getByRole('button', { name: 'Sign up for GitHub' })).toBeVisible();",
            "  });",
            ""
        ]
        
        # Group patterns by type for better organization
        grouped_patterns = self._group_patterns(test_patterns)
        
        # Generate navigation tests (split into smaller groups)
        if grouped_patterns.get('navigation'):
            test_code.extend(self._generate_navigation_tests(grouped_patterns['navigation']))
        
        # Generate form tests
        if grouped_patterns.get('form'):
            test_code.extend(self._generate_form_tests(grouped_patterns['form']))
        
        # Generate search tests
        if grouped_patterns.get('search'):
            test_code.extend(self._generate_search_tests(grouped_patterns['search']))
        
        # Generate assertion tests
        if grouped_patterns.get('assertion'):
            test_code.extend(self._generate_assertion_tests(grouped_patterns['assertion']))
        
        test_code.extend([
            "  test('Error Handling', async () => {",
            "    // Test invalid search",
            "    const searchInput = page.getByPlaceholder(/Search/i);",
            "    await searchInput.fill('!@#$%^&*()');",
            "    await searchInput.press('Enter');",
            "    await expect(page.getByText(/No results|no matches/i)).toBeVisible();",
            "",
            "    // Test invalid navigation",
            f"    await page.goto('{url}/nonexistent-page');",
            "    await expect(page.getByText(/Page not found|404/i)).toBeVisible();",
            "  });",
            "});"
        ])
        
        return '\n'.join(test_code)
    
    def _group_patterns(self, patterns: List[Dict]) -> Dict:
        """Group patterns by element type."""
        grouped = {}
        for pattern in patterns:
            element_type = pattern['element_type']
            if element_type not in grouped:
                grouped[element_type] = []
            grouped[element_type].append(pattern)
        return grouped
    
    def _generate_form_tests(self, form_patterns: List[Dict]) -> List[str]:
        """Generate form interaction test code."""
        code = [
            "  test('Form Elements', async () => {",
            "    // Verify form elements are present and interactive"
        ]
        
        for pattern in form_patterns:
            if pattern['form_type'] == 'authentication':
                code.extend([
                    "    // Sign-in form",
                    "    await expect(page.getByRole('button', { name: 'Sign in' })).toBeVisible();",
                    "    await expect(page.getByRole('link', { name: 'Sign up' })).toBeVisible();",
                    "    await expect(page.getByText('Forgot password?')).toBeVisible();"
                ])
            elif pattern['form_type'] == 'subscription':
                code.extend([
                    "    // Email subscription form",
                    "    const emailInput = page.getByPlaceholder(/Enter.*email/i);",
                    "    await expect(emailInput).toBeVisible();",
                    "    await emailInput.fill('test@example.com');",
                    "    await expect(page.getByRole('button', { name: /Subscribe|Sign up/i })).toBeEnabled();"
                ])
        
        code.extend([
            "  });",
            ""
        ])
        return code
    
    def _generate_search_tests(self, search_patterns: List[Dict]) -> List[str]:
        """Generate search functionality test code."""
        code = [
            "    // Search functionality tests",
            "    const searchInput = page.getByPlaceholder(/Search/i);",
            "    await expect(searchInput).toBeVisible();",
            "    await searchInput.fill('playwright');",
            "    await searchInput.press('Enter');",
            "    await expect(page).toHaveURL(/.*type=repositories/);",
            "    await expect(page.getByText(/repository results/i)).toBeVisible();",
            ""
        ]
        return code
    
    def _generate_navigation_tests(self, navigation_patterns: List[Dict]) -> List[str]:
        """Generate navigation test code."""
        # Group navigation items by section
        sections = {}
        for pattern in navigation_patterns:
            section = pattern.get('section', 'General')
            if section not in sections:
                sections[section] = []
            sections[section].append(pattern)
        
        test_code = []
        
        # Generate separate test for each section
        for section, patterns in sections.items():
            section_name = section if section != 'General' else 'Main'
            test_code.extend([
                f"  test('Navigation - {section_name}', async () => {{",
                "    // Verify navigation links are visible and clickable"
            ])
            
            for pattern in patterns:
                # Clean and escape the link text
                link_text = pattern['text'].replace('\n', ' ').strip()
                link_text = re.sub(r'\s+', ' ', link_text)  # Replace multiple spaces with single space
                link_text = link_text.replace("'", "\\'")  # Escape single quotes
                
                # Generate visibility check
                test_code.append(f"    await expect(page.getByRole('link', {{ name: '{link_text}' }})).toBeVisible();")
            
            test_code.extend([
                "  });",
                ""
            ])
            
            # Generate separate test for actual navigation if needed
            if len(patterns) <= 5:  # Only for sections with few links to avoid long-running tests
                test_code.extend([
                    f"  test('Navigation Click-through - {section_name}', async () => {{",
                ])
                
                for pattern in patterns:
                    link_text = pattern['text'].replace('\n', ' ').strip()
                    link_text = re.sub(r'\s+', ' ', link_text)  # Replace multiple spaces with single space
                    link_text = link_text.replace("'", "\\'")  # Escape single quotes
                    href = pattern['href']
                    
                    # Pre-process the href pattern outside the f-string
                    href_pattern = href.replace('/', r'\\/')
                    
                    test_code.extend([
                        f"    await page.getByRole('link', {{ name: '{link_text}' }}).click();",
                        "    await page.waitForLoadState('networkidle');",
                        f"    await expect(page.url()).toMatch(/{href_pattern}/);",
                        f"    await page.goto('{url}');",  # Go back to homepage
                        "    await page.waitForLoadState('networkidle');",
                        ""
                    ])
                
                test_code.extend([
                    "  });",
                    ""
                ])
        
        return test_code
    
    def _generate_assertion_tests(self, assertion_patterns: List[Dict]) -> List[str]:
        """Generate assertion test code."""
        code = [
            "  test('Content Assertions', async () => {",
            "    // Verify page content"
        ]
        
        for pattern in assertion_patterns:
            if pattern['assertion_type'] == 'header':
                code.extend([
                    f"    // Verify header: {pattern['text']}",
                    f"    await expect(page.getByRole('heading', {{ name: '{pattern['text']}' }})).toBeVisible();"
                ])
            elif pattern['assertion_type'] == 'content':
                # Extract key phrases from content
                text = pattern['text']
                phrases = [t.strip() for t in text.split('\n') if len(t.strip()) > 10][:3]  # Get first 3 substantial phrases
                
                code.extend([
                    "    // Verify main content",
                    *[f"    await expect(page.getByText('{phrase}')).toBeVisible();" for phrase in phrases]
                ])
            
            code.append("")
        
        code.extend([
            "  });",
            ""
        ])
        return code

def main():
    # Initialize analyzer
    analyzer = WebsiteAnalyzer()
    
    # Example usage
    url = "https://example.com"  # Replace with target website
    test_patterns = analyzer.generate_tests(url)
    
    # Print analysis results
    logger.info("\nAnalysis Results:")
    for pattern in test_patterns:
        logger.info(f"Element Type: {pattern['element_type']}")
        logger.info(f"Pattern: {pattern['pattern']}")
        logger.info(f"Confidence: {pattern['confidence']:.4f}")
        logger.info("")

if __name__ == '__main__':
    main() 