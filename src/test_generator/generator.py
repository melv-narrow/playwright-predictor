"""
Test Generator Module

This module generates Playwright test scripts by analyzing web pages
and predicting appropriate test actions using our trained ML model.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from playwright.async_api import Page, ElementHandle
import json
import re
from jinja2 import Environment, FileSystemLoader

from src.prediction.predictor import TestActionPredictor
from src.data_collection.collector import DataCollector

class TestGenerator:
    """
    Generates Playwright test scripts using ML predictions.
    
    This class:
    1. Analyzes web pages to find interactive elements
    2. Predicts appropriate test actions using ML
    3. Generates maintainable Playwright test scripts
    """
    
    def __init__(
        self,
        output_dir: str = "generated_tests",
        model_dir: str = "models",
        template_dir: str = None,
        min_confidence: float = 0.7
    ):
        """
        Initialize test generator.
        
        Args:
            output_dir: Directory to save generated tests
            model_dir: Directory containing trained models
            template_dir: Directory containing test templates
            min_confidence: Minimum confidence threshold for predictions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = Path(model_dir)
        self.min_confidence = min_confidence
        
        # Set up template environment
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
            
        self.template_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Initialize components
        self.collector = DataCollector()
        self.predictor = TestActionPredictor(model_dir=model_dir)
        
        # Set up logging
        logger.add(
            self.output_dir / "generator.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )

    async def generate_test_suite(
        self,
        url: str,
        test_name: str,
        description: Optional[str] = None
    ) -> str:
        """
        Generate a complete test suite for a web page.
        
        Args:
            url: URL of the page to test
            test_name: Name for the generated test
            description: Optional test description
            
        Returns:
            Path to generated test file
        """
        try:
            logger.info(f"Generating test suite for {url}")
            
            # Collect page data
            page_data = await self.collector.collect_page_data(url)
            
            # Generate test steps
            test_steps = await self._generate_test_steps(page_data)
            
            if not test_steps:
                logger.warning("No test steps generated")
                return None
            
            # Create test file
            test_file = self._generate_test_file(
                test_name=test_name,
                url=url,
                description=description,
                test_steps=test_steps
            )
            
            logger.info(f"Generated test suite: {test_file}")
            return test_file
            
        except Exception as e:
            logger.error(f"Test generation failed: {str(e)}")
            raise
            
    async def _generate_test_steps(
        self,
        page_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate test steps from page data.
        
        Args:
            page_data: Collected page data
            
        Returns:
            List of test steps
        """
        test_steps = []
        
        for element_data in page_data['elements']:
            # Get prediction
            prediction = self.predictor.predict_action(element_data)
            
            if prediction['confidence'] < self.min_confidence:
                continue
                
            # Create test step
            test_step = self._generate_test_step(element_data, prediction)
            
            if test_step is not None:
                test_steps.append(test_step)
                
        return test_steps
        
    def _generate_test_step(self, element_data: Dict[str, Any], prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a test step based on element data and prediction.
        
        Args:
            element_data: Element data dictionary
            prediction: Prediction dictionary with action and confidence
            
        Returns:
            Test step dictionary or None if step should be skipped
        """
        try:
            tag_name = element_data.get('tag_name', '').lower()
            element_type = element_data.get('attributes', {}).get('type', '').lower()
            
            # Generate reliable selector
            selector = self._generate_reliable_selector(element_data)
            if not selector:
                return None
                
            # Determine appropriate action based on element type
            action = prediction['action']
            value = None
            
            if tag_name == 'input':
                if element_type in ['text', 'email', 'password', 'search', 'tel', 'url']:
                    action = 'fill'
                    value = self._get_default_value(element_type)
                elif element_type in ['submit', 'button', 'reset']:
                    action = 'click'
                elif element_type in ['checkbox', 'radio']:
                    action = 'check'
            elif tag_name == 'button':
                action = 'click'
            elif tag_name == 'a':
                action = 'click'
            elif tag_name == 'select':
                action = 'select_option'
                value = '0'  # Select first option by default
            elif tag_name == 'textarea':
                action = 'fill'
                value = 'Test input'
                
            # Create test step
            test_step = {
                'action': action,
                'selector': selector,
                'confidence': prediction['confidence']
            }
            
            if value is not None:
                test_step['value'] = value
                
            return test_step
            
        except Exception as e:
            logger.error(f"Failed to generate test step: {str(e)}")
            return None
            
    def _get_default_value(self, input_type: str) -> str:
        """Get default test value based on input type."""
        defaults = {
            'email': 'test@example.com',
            'password': 'TestPassword123',
            'search': 'test search',
            'tel': '1234567890',
            'url': 'https://example.com',
            'text': 'Test input'
        }
        return defaults.get(input_type, 'Test input')
        
    def _generate_test_file(
        self,
        test_name: str,
        url: str,
        description: Optional[str],
        test_steps: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a Playwright test file.
        
        Args:
            test_name: Name of the test
            url: Target URL
            description: Test description
            test_steps: List of test steps
            
        Returns:
            Path to generated test file
        """
        try:
            # Load template
            template = self.template_env.get_template("test_template.py.jinja")
            
            # Clean and sanitize test steps
            cleaned_steps = []
            for step in test_steps:
                # Clean any special characters from selectors and values
                cleaned_step = {
                    'action': step['action'],
                    'selector': self._clean_string(step['selector']),
                    'confidence': step['confidence']
                }
                if 'value' in step:
                    cleaned_step['value'] = self._clean_string(step['value'])
                cleaned_steps.append(cleaned_step)
            
            # Generate test content
            test_content = template.render(
                test_name=test_name,
                url=url,
                description=description or f"Generated test for {url}",
                test_steps=cleaned_steps,
                timestamp=self._get_timestamp()
            )
            
            # Save test file
            test_file = self.output_dir / f"{self._sanitize_filename(test_name)}_test.py"
            test_file.write_text(test_content, encoding='utf-8')
            
            return str(test_file)
            
        except Exception as e:
            logger.error(f"Failed to generate test file: {str(e)}")
            raise
            
    def _clean_string(self, text: str) -> str:
        """Clean string of special characters and encode properly."""
        if not text:
            return ""
            
        # Replace common special characters
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...' # Ellipsis
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Remove any other non-ASCII characters
        text = ''.join(char for char in text if ord(char) < 128)
        
        return text
        
    def _generate_reliable_selector(
        self,
        element_data: Dict[str, Any]
    ) -> str:
        """
        Generate a reliable CSS selector for an element.
        
        Args:
            element_data: Element data
            
        Returns:
            CSS selector string
        """
        selectors = []
        
        # Try ID
        if 'id' in element_data.get('attributes', {}):
            selectors.append(f"#{element_data['attributes']['id']}")
            
        # Try test-id
        if 'data-testid' in element_data.get('attributes', {}):
            selectors.append(
                f"[data-testid='{element_data['attributes']['data-testid']}']"
            )
            
        # Try name
        if 'name' in element_data.get('attributes', {}):
            selectors.append(f"[name='{element_data['attributes']['name']}']")
            
        # Use tag with classes as fallback
        if 'class' in element_data.get('attributes', {}):
            classes = element_data['attributes']['class'].split()
            class_selector = '.'.join(classes)
            selectors.append(f"{element_data['tag_name']}.{class_selector}")
            
        # Use tag with text as last resort
        if element_data.get('inner_text'):
            text = element_data['inner_text'].strip()
            if text:
                selectors.append(
                    f"{element_data['tag_name']}:text('{text}')"
                )
                
        return selectors[0] if selectors else element_data.get(
            'selector',
            f"{element_data['tag_name']}"
        )
        
    def _sanitize_filename(self, filename: str) -> str:
        """Convert string to valid filename."""
        # Replace invalid characters
        filename = re.sub(r'[^\w\s-]', '', filename)
        # Replace spaces with underscores
        return re.sub(r'[-\s]+', '_', filename).strip('-_')
        
    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
