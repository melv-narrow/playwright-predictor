"""
Template-based Test Generator Module

This module generates Playwright test scripts using predefined templates
and website analysis.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import re

from src.data_collection.collector import DataCollector

class TemplateBasedGenerator:
    """
    Generates Playwright test scripts using predefined templates.
    """
    
    def __init__(
        self,
        output_dir: str = "generated_tests",
        template_dir: str = None
    ):
        """
        Initialize template-based generator.
        
        Args:
            output_dir: Directory to save generated tests
            template_dir: Directory containing test templates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up template environment
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
            
        self.template_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Initialize collector
        self.collector = DataCollector()
        
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
        description: Optional[str] = None,
        template_name: str = "sliq_marketing_template.py.jinja"
    ) -> str:
        """
        Generate a test suite using a template.
        
        Args:
            url: URL of the page to test
            test_name: Name for the generated test
            description: Optional test description
            template_name: Name of the template to use
            
        Returns:
            Path to generated test file
        """
        try:
            logger.info(f"Generating test suite for {url}")
            
            # Collect page data
            page_data = await self.collector.collect_page_data(url)
            
            # Analyze page structure
            template_data = await self._analyze_page_structure(page_data)
            template_data.update({
                'base_url': url,
                'timestamp': self._get_timestamp()
            })
            
            # Generate test file from template
            test_file = self._generate_test_file(
                test_name=test_name,
                template_name=template_name,
                template_data=template_data,
                description=description
            )
            
            logger.info(f"Generated test suite: {test_file}")
            return test_file
            
        except Exception as e:
            logger.error(f"Test generation failed: {str(e)}")
            raise

    async def _analyze_page_structure(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze page structure to populate template data.
        
        Args:
            page_data: Collected page data
            
        Returns:
            Dictionary of template variables
        """
        # Extract navigation items
        nav_items = []
        service_categories = []
        contact_form_elements = []
        has_blog = False
        has_testimonials = False
        has_portfolio = False
        
        for element in page_data['elements']:
            # Navigation items
            if element.get('role') == 'navigation':
                for link in element.get('children', []):
                    if link.get('tag_name') == 'a':
                        nav_items.append(link.get('text', '').strip())
            
            # Service categories
            if 'service' in element.get('text', '').lower():
                service_categories.append(element.get('text', '').strip())
            
            # Contact form
            if element.get('tag_name') == 'form':
                for input_el in element.get('children', []):
                    if input_el.get('tag_name') == 'input':
                        contact_form_elements.append({
                            'selector': self._generate_selector(input_el),
                            'type': input_el.get('attributes', {}).get('type', 'text')
                        })
            
            # Check for blog
            if 'blog' in element.get('text', '').lower():
                has_blog = True
            
            # Check for testimonials
            if 'testimonial' in element.get('text', '').lower():
                has_testimonials = True
            
            # Check for portfolio
            if any(word in element.get('text', '').lower() for word in ['portfolio', 'work', 'projects']):
                has_portfolio = True
        
        return {
            'navigation_items': list(set(nav_items)),
            'service_categories': list(set(service_categories)),
            'contact_form_elements': contact_form_elements,
            'has_blog': has_blog,
            'has_testimonials': has_testimonials,
            'has_portfolio': has_portfolio
        }

    def _generate_test_file(
        self,
        test_name: str,
        template_name: str,
        template_data: Dict[str, Any],
        description: Optional[str] = None
    ) -> Path:
        """
        Generate a test file from template.
        
        Args:
            test_name: Name of the test
            template_name: Name of the template to use
            template_data: Data to populate template
            description: Optional test description
            
        Returns:
            Path to generated test file
        """
        # Load template
        template = self.template_env.get_template(template_name)
        
        # Add description to template data
        if description:
            template_data['description'] = description
        
        # Render template
        test_content = template.render(**template_data)
        
        # Save to file
        test_file = self.output_dir / f"{self._sanitize_filename(test_name)}.py"
        test_file.write_text(test_content)
        
        return test_file

    def _generate_selector(self, element_data: Dict[str, Any]) -> str:
        """Generate a reliable CSS selector for an element."""
        selectors = []
        
        # Use ID if available
        if element_id := element_data.get('attributes', {}).get('id'):
            return f"#{element_id}"
        
        # Use combination of tag and classes
        tag_name = element_data.get('tag_name', '')
        if classes := element_data.get('attributes', {}).get('class', '').split():
            return f"{tag_name}.{'.'.join(classes)}"
        
        # Use tag name and attributes
        if attributes := element_data.get('attributes', {}):
            for key, value in attributes.items():
                if key not in ['class', 'style']:
                    selectors.append(f"[{key}='{value}']")
        
        return tag_name + ''.join(selectors)

    def _sanitize_filename(self, filename: str) -> str:
        """Convert string to valid filename."""
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        return filename.lower()

    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
