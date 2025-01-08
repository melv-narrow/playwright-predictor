import asyncio
from playwright.async_api import async_playwright
import argparse
from pathlib import Path
import json
import logging
from typing import List, Dict, Any
import sys
from generate_playwright_test import generate_test_case, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def analyze_page_elements(page) -> List[Dict[str, Any]]:
    """Analyze page elements and collect their properties."""
    elements = []
    
    # Get all interactive elements
    selectors = [
        "button", "a", "input", "select", "textarea",
        "[role='button']", "[role='link']", "[role='textbox']",
        "form", "[data-testid]", "[aria-label]"
    ]
    
    for selector in selectors:
        try:
            elements_found = await page.query_selector_all(selector)
            for element in elements_found:
                try:
                    # Get element properties
                    element_info = {
                        "tag_name": await element.evaluate("el => el.tagName.toLowerCase()"),
                        "type": await element.get_attribute("type"),
                        "id": await element.get_attribute("id"),
                        "class": await element.get_attribute("class"),
                        "name": await element.get_attribute("name"),
                        "value": await element.get_attribute("value"),
                        "placeholder": await element.get_attribute("placeholder"),
                        "href": await element.get_attribute("href"),
                        "role": await element.get_attribute("role"),
                        "aria-label": await element.get_attribute("aria-label"),
                        "data-testid": await element.get_attribute("data-testid"),
                        "selector": selector,
                        "text": await element.text_content(),
                        "is_visible": await element.is_visible(),
                        "is_enabled": await element.is_enabled(),
                    }
                    
                    # Clean up None values
                    element_info = {k: v for k, v in element_info.items() if v is not None}
                    elements.append(element_info)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing element: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error with selector {selector}: {e}")
            continue
            
    return elements

async def crawl_website(url: str, max_pages: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Crawl website and collect information about pages and their elements."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()
        
        pages_analyzed = {}
        pages_to_visit = {url}
        visited_pages = set()
        
        try:
            while pages_to_visit and len(visited_pages) < max_pages:
                current_url = pages_to_visit.pop()
                if current_url in visited_pages:
                    continue
                    
                logger.info(f"Analyzing page: {current_url}")
                
                try:
                    await page.goto(current_url, wait_until="networkidle")
                    visited_pages.add(current_url)
                    
                    # Analyze page elements
                    elements = await analyze_page_elements(page)
                    pages_analyzed[current_url] = elements
                    
                    # Find links to other pages on the same domain
                    base_domain = current_url.split("/")[2]
                    links = await page.query_selector_all("a")
                    
                    for link in links:
                        href = await link.get_attribute("href")
                        if href and href.startswith(("http://", "https://")):
                            link_domain = href.split("/")[2]
                            if link_domain == base_domain:
                                pages_to_visit.add(href)
                                
                except Exception as e:
                    logger.error(f"Error analyzing page {current_url}: {e}")
                    continue
                    
        finally:
            await browser.close()
            
        return pages_analyzed

def generate_tests_for_website(url: str, output_dir: str, max_pages: int = 5):
    """Generate Playwright tests for a website."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the model
    model = load_model()
    
    # Crawl website and collect data
    pages_data = asyncio.run(crawl_website(url, max_pages))
    
    # Generate tests for each page
    for page_url, elements in pages_data.items():
        logger.info(f"Generating tests for page: {page_url}")
        
        # Generate test file name from URL
        page_name = page_url.split("/")[-1]
        if not page_name:
            page_name = "index"
        test_file = output_path / f"test_{page_name}.spec.ts"
        
        # Generate test cases
        test_cases = []
        for element in elements:
            try:
                test_case = generate_test_case(model, element)
                if test_case:
                    test_cases.append(test_case)
            except Exception as e:
                logger.warning(f"Error generating test case for element: {e}")
                continue
        
        # Write test file
        if test_cases:
            test_content = f"""
import {{ test, expect }} from '@playwright/test';

test.describe('{page_url}', () => {{
    test.beforeEach(async ({{ page }}) => {{
        await page.goto('{page_url}');
    }});
    
    {chr(10).join(test_cases)}
}});
"""
            test_file.write_text(test_content)
            logger.info(f"Generated test file: {test_file}")
            
def main():
    parser = argparse.ArgumentParser(description="Generate Playwright tests for a website")
    parser.add_argument("url", help="Website URL to generate tests for")
    parser.add_argument("--output", "-o", default="./tests",
                      help="Output directory for generated tests")
    parser.add_argument("--max-pages", "-m", type=int, default=5,
                      help="Maximum number of pages to analyze")
    
    args = parser.parse_args()
    
    try:
        generate_tests_for_website(args.url, args.output, args.max_pages)
    except Exception as e:
        logger.error(f"Error generating tests: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
