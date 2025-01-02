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
        model_dir: str = "models/rf",
        confidence_threshold: float = 0.8,
        template_dir: str = "src/test_generation/templates"
    ):
        """
        Initialize the test generator.
        
        Args:
            output_dir: Directory to save generated tests
            model_dir: Directory containing trained model
            confidence_threshold: Minimum confidence for predictions
            template_dir: Directory containing test templates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize predictor and collector
        self.predictor = TestActionPredictor(model_dir=model_dir)
        self.collector = DataCollector()
        
        self.confidence_threshold = confidence_threshold
        
        # Set up Jinja2 for test templates
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Set up logging
        logger.add(
            self.output_dir / "generation.log",
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
            
            if prediction['confidence'] < self.confidence_threshold:
                continue
                
            # Create test step
            test_step = {
                'action': prediction['action'],
                'selector': self._generate_reliable_selector(element_data),
                'value': self._generate_test_value(
                    element_data,
                    prediction['action']
                ),
                'confidence': prediction['confidence']
            }
            
            test_steps.append(test_step)
            
        return test_steps
        
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
                    'value': self._clean_string(step.get('value', '')),
                    'confidence': step['confidence']
                }
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
        
    def _generate_test_value(
        self,
        element_data: Dict[str, Any],
        action: str
    ) -> Optional[str]:
        """
        Generate appropriate test value for an action.
        
        Args:
            element_data: Element data
            action: Predicted action
            
        Returns:
            Test value if needed
        """
        if action != 'type':
            return None
            
        # Generate appropriate test value based on input type
        input_type = element_data.get('attributes', {}).get('type', 'text')
        
        if input_type == 'email':
            return 'test@example.com'
        elif input_type == 'password':
            return 'TestPassword123!'
        elif input_type == 'number':
            return '12345'
        elif input_type == 'tel':
            return '+1234567890'
        elif input_type == 'search':
            return 'test search'
        else:
            # Generate from placeholder or name
            placeholder = element_data.get('attributes', {}).get('placeholder', '')
            name = element_data.get('attributes', {}).get('name', '')
            
            if 'email' in (placeholder.lower() + name.lower()):
                return 'test@example.com'
            elif 'password' in (placeholder.lower() + name.lower()):
                return 'TestPassword123!'
            elif 'phone' in (placeholder.lower() + name.lower()):
                return '+1234567890'
            else:
                return 'Test Input'
                
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
