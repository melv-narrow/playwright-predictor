"""
Data Collection Module

This module handles collecting training data from web pages, including
element attributes, actions, and page structure.
"""

import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
from loguru import logger
from playwright.async_api import (
    async_playwright,
    Page,
    ElementHandle,
    Browser,
    BrowserContext
)
import pandas as pd
from urllib.parse import urlparse, urljoin
import os

class DataCollector:
    """
    A class for collecting training data from websites.
    
    This class uses Playwright to navigate websites and collect data about
    interactive elements that could be useful for test generation.
    
    Attributes:
        output_dir (str): Directory to save collected data
        headless (bool): Whether to run browser in headless mode
        viewport_size (Dict[str, int]): Browser viewport dimensions
        page_timeout (int): Page load timeout in milliseconds
        max_retries (int): Maximum number of retries for failed operations
        retry_delay (int): Delay between retries in milliseconds
        visited_urls (Set[str]): Set of URLs already visited
        interactive_selectors (Set[str]): CSS selectors for interactive elements
        max_elements (int): Maximum elements to collect per page
    """
    
    def __init__(
        self,
        output_dir: str = "./training_data",
        max_elements: int = 1000,
        headless: bool = False,
        viewport_size: Dict[str, int] = {"width": 1280, "height": 720},
        page_timeout: int = 60000,  # 60 seconds
        max_retries: int = 3,
        retry_delay: int = 5000  # 5 seconds
    ):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Directory to save collected data
            max_elements: Maximum elements to collect per page
            headless: Whether to run browser in headless mode
            viewport_size: Browser viewport dimensions
            page_timeout: Page load timeout in milliseconds
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries in milliseconds
        """
        # Initialize configuration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_elements = max_elements
        self.headless = headless
        self.viewport_size = viewport_size
        self.page_timeout = page_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize state
        self.visited_urls: set[str] = set()
        
        # Define selectors for interactive elements
        self.interactive_selectors = {
            "button",
            "a[href]",
            "input",
            "select",
            "textarea",
            "[role='button']",
            "[role='link']",
            "[role='checkbox']",
            "[role='radio']",
            "[role='tab']",
            "[role='menuitem']",
            "[role='listbox']",
            "[role='combobox']",
            "[role='switch']",
            "[role='searchbox']",
            "form",
            "label"
        }
        
        # Set up logging
        logger.add(
            f"{self.output_dir}/collector.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
    async def collect_from_url(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 10
    ) -> str:
        """
        Collect training data from a website.
        
        This method will:
        1. Start a browser session
        2. Visit the given URL
        3. Collect data about interactive elements
        4. Follow links up to max_depth
        5. Save collected data
        
        Args:
            url: Starting URL
            max_depth: Maximum crawl depth
            max_pages: Maximum number of pages to visit
            
        Returns:
            Path to the saved dataset
            
        Raises:
            Exception: If data collection fails
        """
        try:
            logger.info(f"Starting data collection from {url}")
            self.visited_urls.clear()
            collected_data: List[Dict[str, Any]] = []
            
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context(
                    viewport=self.viewport_size,
                    record_har_path=f"{self.output_dir}/trace.har"
                )
                
                # Configure timeouts
                context.set_default_timeout(self.page_timeout)
                context.set_default_navigation_timeout(self.page_timeout)
                
                # Start collection
                page = await context.new_page()
                await self._collect_from_page(
                    page=page,
                    url=url,
                    collected_data=collected_data,
                    current_depth=0,
                    max_depth=max_depth,
                    max_pages=max_pages
                )
                
                await browser.close()
                logger.info("Browser session closed")
                
            # Save and return dataset path
            return await self._save_dataset(collected_data)
            
        except Exception as e:
            logger.error(f"Data collection failed: {str(e)}")
            raise
            
    async def _collect_from_page(
        self,
        page: Page,
        url: str,
        collected_data: List[Dict[str, Any]],
        current_depth: int,
        max_depth: int,
        max_pages: int
    ) -> None:
        """
        Collect data from a single page and its links.
        
        Args:
            page: Playwright page object
            url: URL to collect from
            collected_data: List to store collected data
            current_depth: Current crawl depth
            max_depth: Maximum crawl depth
            max_pages: Maximum number of pages to visit
        """
        if (
            url in self.visited_urls or
            len(self.visited_urls) >= max_pages or
            current_depth > max_depth
        ):
            return
            
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Collecting from {url} (depth: {current_depth})")
                
                # Visit page with retry logic
                await page.goto(url, wait_until='domcontentloaded')
                await asyncio.sleep(2)  # Short wait for dynamic content
                
                try:
                    await page.wait_for_load_state('networkidle', timeout=10000)
                except TimeoutError:
                    logger.warning(f"Network did not become idle on {url}, continuing anyway")
                
                self.visited_urls.add(url)
                
                # Collect interactive elements
                elements_data = await self._collect_page_elements(page)
                collected_data.extend(elements_data)
                
                # Find links for crawling
                if current_depth < max_depth:
                    links = await self._extract_links(page, url)
                    for link in links:
                        if len(self.visited_urls) >= max_pages:
                            break
                        await self._collect_from_page(
                            page=page,
                            url=link,
                            collected_data=collected_data,
                            current_depth=current_depth + 1,
                            max_depth=max_depth,
                            max_pages=max_pages
                        )
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                    await asyncio.sleep(self.retry_delay / 1000)  # Convert ms to seconds
                else:
                    logger.error(f"All attempts failed for {url}: {str(e)}")
            
    async def _collect_page_elements(self, page: Page) -> List[Dict[str, Any]]:
        """
        Collect data about interactive elements on the page.
        
        This method:
        1. Finds all interactive elements using predefined selectors
        2. Extracts their properties and attributes
        3. Determines appropriate test actions
        4. Generates unique selectors for each element
        
        Args:
            page: Playwright page object
            
        Returns:
            List of dictionaries containing element data
        """
        elements_data: List[Dict[str, Any]] = []
        
        for selector in self.interactive_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements[:self.max_elements // len(self.interactive_selectors)]:
                    try:
                        # Get element properties
                        data = {
                            'url': page.url,
                            'html': await element.evaluate('(el) => el.outerHTML'),
                            'tag_name': await element.evaluate('(el) => el.tagName.toLowerCase()'),
                            'attributes': await element.evaluate('''(el) => {
                                const attrs = {};
                                for (const attr of el.attributes) {
                                    attrs[attr.name] = attr.value;
                                }
                                return attrs;
                            }'''),
                            'inner_text': await element.inner_text(),
                            'is_visible': await element.is_visible(),
                            'bounding_box': await element.bounding_box(),
                            'selector': await element.evaluate('''(el) => {
                                const path = [];
                                let node = el;
                                while (node && node.parentNode) {
                                    if (node.id) {
                                        path.unshift(`#${node.id}`);
                                        break;
                                    }
                                    const siblings = Array.from(node.parentNode.children);
                                    const index = siblings.indexOf(node) + 1;
                                    path.unshift(`${node.tagName.toLowerCase()}:nth-child(${index})`);
                                    node = node.parentNode;
                                    if (node.id) {
                                        path.unshift(`#${node.id}`);
                                        break;
                                    }
                                }
                                return path.join(" > ");
                            }''')
                        }
                        
                        # Determine element type and action
                        data['suggested_action'] = self._determine_element_action(data)
                        elements_data.append(data)
                        
                    except Exception as e:
                        logger.warning(f"Failed to collect element data: {str(e)}")
                        
            except Exception as e:
                logger.warning(f"Failed to query selector {selector}: {str(e)}")
                
        return elements_data
        
    def _determine_element_action(self, element_data: Dict[str, Any]) -> str:
        """
        Determine appropriate test action for an element.
        
        This method analyzes the element's properties to suggest
        the most appropriate test action (click, input, check, etc.)
        
        Args:
            element_data: Dictionary containing element properties
            
        Returns:
            Suggested test action as string
        """
        tag = element_data['tag_name']
        attrs = element_data['attributes']
        
        if tag == 'button' or 'button' in attrs.get('role', ''):
            return 'click'
        elif tag == 'input':
            input_type = attrs.get('type', 'text')
            if input_type in ['submit', 'button']:
                return 'click'
            elif input_type in ['checkbox', 'radio']:
                return 'check'
            elif input_type in ['number', 'date', 'time']:
                return 'fill'
            else:
                return 'type'
        elif tag == 'select':
            return 'select'
        elif tag == 'textarea':
            return 'fill'
        elif tag == 'a' or 'link' in attrs.get('role', ''):
            return 'click'
        elif tag == 'form':
            return 'submit'
        else:
            return 'assert'
            
    async def _extract_links(self, page: Page, base_url: str) -> List[str]:
        """
        Extract valid links from the page for crawling.
        
        This method:
        1. Finds all anchor tags with href attributes
        2. Converts relative URLs to absolute
        3. Filters to only include links from the same domain
        
        Args:
            page: Playwright page object
            base_url: Base URL for resolving relative links
            
        Returns:
            List of valid URLs to crawl
        """
        links = set()
        base_domain = urlparse(base_url).netloc
        
        try:
            # Get all links on the page
            elements = await page.query_selector_all('a[href]')
            for element in elements:
                try:
                    href = await element.get_attribute('href')
                    if href:
                        absolute_url = urljoin(base_url, href)
                        parsed = urlparse(absolute_url)
                        # Only include links to the same domain
                        if parsed.netloc == base_domain:
                            links.add(absolute_url)
                except Exception as e:
                    logger.warning(f"Failed to extract link: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"Failed to query links: {str(e)}")
            
        return list(links)
        
    async def _save_dataset(self, collected_data: List[Dict[str, Any]]) -> str:
        """
        Save collected data to disk in multiple formats.
        
        This method saves:
        1. Raw JSON data for maximum fidelity
        2. CSV format for easy analysis
        3. Includes timestamp in filenames
        
        Args:
            collected_data: List of dictionaries containing element data
            
        Returns:
            Path to the saved CSV dataset
            
        Raises:
            Exception: If saving fails
        """
        try:
            # Save raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = f"{self.output_dir}/raw_data_{timestamp}.json"
            with open(raw_path, 'w') as f:
                json.dump(collected_data, f, indent=2)
                
            # Convert to pandas DataFrame for easier processing
            df = pd.DataFrame(collected_data)
            csv_path = f"{self.output_dir}/dataset_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Dataset saved to {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            raise

    async def collect_page_data(
        self,
        url: str,
        wait_for_load: bool = True,
        timeout: int = 30000
    ) -> Dict[str, Any]:
        """
        Collect data from a web page.
        
        Args:
            url: URL to collect data from
            wait_for_load: Whether to wait for page load
            timeout: Page load timeout in ms
            
        Returns:
            Dictionary containing page data and elements
        """
        try:
            logger.info(f"Collecting data from {url}")
            
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context()
                page = await context.new_page()
                
                # Navigate to page
                await page.goto(url)
                if wait_for_load:
                    await page.wait_for_load_state("networkidle")
                    
                # Collect page data
                page_data = {
                    'url': url,
                    'title': await page.title(),
                    'elements': await self._collect_elements(page)
                }
                
                await browser.close()
                return page_data
                
        except Exception as e:
            logger.error(f"Failed to collect page data: {str(e)}")
            raise
            
    async def _collect_elements(self, page: Page) -> List[Dict[str, Any]]:
        """
        Collect data about interactive elements on the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of element data dictionaries
        """
        elements_data = []
        
        # Find interactive elements
        selectors = [
            'a[href]',  # Links
            'button',   # Buttons
            'input',    # Input fields
            'form',     # Forms
            'select',   # Dropdowns
            '[role="button"]',  # ARIA buttons
            '[role="link"]',    # ARIA links
            '[onclick]',        # Elements with click handlers
            '[tabindex]'        # Focusable elements
        ]
        
        for selector in selectors:
            elements = await page.query_selector_all(selector)
            
            for element in elements[:self.max_elements // len(selectors)]:
                try:
                    element_data = await self._extract_element_data(element)
                    if element_data:
                        elements_data.append(element_data)
                except Exception as e:
                    logger.warning(f"Failed to extract element data: {str(e)}")
                    continue
                    
        return elements_data
        
    async def _extract_element_data(
        self,
        element: ElementHandle
    ) -> Optional[Dict[str, Any]]:
        """
        Extract relevant data from an element.
        
        Args:
            element: Playwright element handle
            
        Returns:
            Dictionary of element data
        """
        try:
            # Get element properties
            properties = await element.evaluate("""element => {
                const rect = element.getBoundingClientRect();
                return {
                    tagName: element.tagName,
                    id: element.id,
                    className: element.className,
                    innerText: element.innerText,
                    value: element.value,
                    type: element.type,
                    href: element.href,
                    placeholder: element.placeholder,
                    name: element.name,
                    isVisible: element.offsetParent !== null,
                    attributes: Array.from(element.attributes).reduce((obj, attr) => {
                        obj[attr.name] = attr.value;
                        return obj;
                    }, {}),
                    boundingBox: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    }
                };
            }""")
            
            # Get element selector
            selector = await element.evaluate("""element => {
                const getSelector = el => {
                    if (el.id) return `#${el.id}`;
                    if (el.className) {
                        const classes = Array.from(el.classList).join('.');
                        return `${el.tagName.toLowerCase()}.${classes}`;
                    }
                    return el.tagName.toLowerCase();
                };
                return getSelector(element);
            }""")
            
            return {
                'tag_name': properties['tagName'].lower(),
                'attributes': properties['attributes'],
                'inner_text': properties['innerText'],
                'is_visible': properties['isVisible'],
                'bounding_box': properties['boundingBox'],
                'selector': selector
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract element properties: {str(e)}")
            return None
