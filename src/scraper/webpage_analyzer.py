"""
Webpage Analyzer

This module handles webpage analysis and data collection for training the ML model.
It extracts relevant HTML elements and their properties using Playwright.
"""

import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from loguru import logger
from typing import Dict, List, Optional, Set
import json
import os

class WebpageAnalyzer:
    def __init__(
        self,
        headless: bool = True,
        viewport_size: Dict[str, int] = {"width": 1280, "height": 720}
    ):
        """
        Initialize the webpage analyzer.
        
        Args:
            headless: Whether to run browser in headless mode
            viewport_size: Browser viewport dimensions
        """
        self.headless = headless
        self.viewport_size = viewport_size
        self.interactive_elements: Set[str] = {
            "a", "button", "input", "select", "textarea", "form"
        }
        
    async def analyze_page(
        self,
        url: str,
        wait_for_load: bool = True,
        timeout: int = 30000
    ) -> Dict:
        """
        Analyze a webpage and extract relevant elements.
        
        Args:
            url: URL of the webpage to analyze
            wait_for_load: Whether to wait for page load state
            timeout: Maximum time to wait for page load in ms
            
        Returns:
            Dictionary containing extracted page data
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context(
                    viewport=self.viewport_size
                )
                page = await context.new_page()
                
                # Navigate to page
                response = await page.goto(url, timeout=timeout)
                if not response:
                    raise Exception(f"Failed to load {url}")
                    
                if wait_for_load:
                    await page.wait_for_load_state("networkidle")
                
                # Extract page content
                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                
                # Collect interactive elements
                elements_data = []
                for tag in self.interactive_elements:
                    elements = soup.find_all(tag)
                    for elem in elements:
                        element_data = {
                            "tag": tag,
                            "html": str(elem),
                            "attributes": dict(elem.attrs),
                            "text_content": elem.get_text(strip=True),
                            "xpath": self._get_xpath(elem),
                            "css_selector": self._get_css_selector(elem)
                        }
                        elements_data.append(element_data)
                
                # Collect page metadata
                page_data = {
                    "url": url,
                    "title": await page.title(),
                    "elements": elements_data
                }
                
                await browser.close()
                return page_data
                
        except Exception as e:
            logger.error(f"Failed to analyze {url}: {str(e)}")
            raise
            
    def _get_xpath(self, element) -> str:
        """Generate XPath for a BeautifulSoup element."""
        components = []
        child = element
        for parent in element.parents:
            siblings = parent.find_all(child.name, recursive=False)
            if len(siblings) == 1:
                components.append(child.name)
            else:
                index = siblings.index(child) + 1
                components.append(f"{child.name}[{index}]")
            child = parent
        components.reverse()
        return "/" + "/".join(components)
        
    def _get_css_selector(self, element) -> str:
        """Generate CSS selector for a BeautifulSoup element."""
        if "id" in element.attrs:
            return f"#{element['id']}"
        elif "class" in element.attrs:
            return f".{'.'.join(element['class'])}"
        else:
            return self._get_xpath(element)
            
    async def save_analysis(
        self,
        url: str,
        output_dir: str,
        filename: Optional[str] = None
    ) -> None:
        """
        Analyze a webpage and save results to disk.
        
        Args:
            url: URL to analyze
            output_dir: Directory to save results
            filename: Optional custom filename
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            if filename is None:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                filename = f"{parsed.netloc.replace('.', '_')}.json"
                
            analysis = await self.analyze_page(url)
            
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w") as f:
                json.dump(analysis, f, indent=2)
                
            logger.info(f"Saved analysis to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")
            raise
