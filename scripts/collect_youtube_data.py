"""
Collect training data from YouTube's interface for AI test generation.
Handles dynamic content and extracts semantic features specific to video platforms.
"""

from playwright.sync_api import sync_playwright, Page
import pandas as pd
import json
from pathlib import Path
import time
from typing import Dict, List, Any
import logging

# Create logging directory
log_dir = Path('training_data/youtube')
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_dir / 'collector.log'
)

class YouTubeDataCollector:
    def __init__(self):
        self.data = []
        self.output_dir = Path('training_data/youtube')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_element_data(self, element: Any, page: Page) -> Dict:
        """Extract relevant data from a DOM element."""
        try:
            box = element.bounding_box()
            attrs = element.evaluate('el => Object.assign({}, ...Array.from(el.attributes, ({name, value}) => ({[name]: value})))')
            
            return {
                'url': page.url,
                'tag_name': element.evaluate('el => el.tagName.toLowerCase()'),
                'attributes': attrs,
                'inner_text': element.inner_text().strip(),
                'is_visible': element.is_visible(),
                'bounding_box': box,
                'selector': element.evaluate('el => cssPath(el)'), # Custom cssPath function injected
                'aria_role': element.get_attribute('role'),
                'aria_label': element.get_attribute('aria-label'),
                'classes': element.get_attribute('class'),
                'id': element.get_attribute('id')
            }
        except Exception as e:
            logging.error(f"Error extracting element data: {str(e)}")
            return None

    def inject_helper_functions(self, page: Page):
        """Inject helper functions for element analysis."""
        # Inject cssPath function for reliable selector generation
        page.evaluate('''() => {
            window.cssPath = function(el) {
                if (!(el instanceof Element)) return;
                var path = [];
                while (el.nodeType === Node.ELEMENT_NODE) {
                    var selector = el.nodeName.toLowerCase();
                    if (el.id) {
                        selector += '#' + el.id;
                        path.unshift(selector);
                        break;
                    } else {
                        var sib = el, nth = 1;
                        while (sib.previousElementSibling) {
                            sib = sib.previousElementSibling;
                            if (sib.nodeName.toLowerCase() == selector)
                               nth++;
                        }
                        if (nth != 1)
                            selector += ":nth-of-type("+nth+")";
                    }
                    path.unshift(selector);
                    el = el.parentNode;
                }
                return path.join(" > ");
            }
        }''')

    async def collect_interaction_data(self, page: Page):
        """Collect data about user interactions."""
        # Monitor click events
        page.on('click', lambda event: self.handle_interaction('click', event))
        
        # Monitor scroll events
        await page.evaluate('''() => {
            window.addEventListener('scroll', (e) => {
                // Custom scroll event handling
            });
        }''')

    def analyze_video_specific_elements(self, page: Page):
        """Analyze video-specific UI elements."""
        selectors = [
            'video',
            '[aria-label*="video"]',
            '[role="button"]',
            'ytd-thumbnail',
            'ytd-video-renderer',
            '#movie_player',
            '.ytp-chrome-controls'
        ]
        
        video_elements = []
        for selector in selectors:
            try:
                elements = page.query_selector_all(selector)
                for element in elements:
                    data = self.extract_element_data(element, page)
                    if data:
                        data['element_type'] = 'video_control'
                        video_elements.append(data)
            except Exception as e:
                logging.error(f"Error analyzing video elements: {str(e)}")
        
        return video_elements

    def collect_data(self):
        """Main method to collect training data from YouTube."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            
            try:
                # Navigate to YouTube
                logging.info("Navigating to YouTube...")
                page.goto('https://www.youtube.com')
                time.sleep(5)  # Allow dynamic content to load
                
                # Inject helper functions
                self.inject_helper_functions(page)
                
                # Collect general interface elements
                logging.info("Collecting interface elements...")
                selectors = [
                    'button', 'a', 'input', 'ytd-thumbnail',
                    '[role="button"]', '[role="tab"]',
                    '#content', '#container', '.style-scope'
                ]
                
                for selector in selectors:
                    elements = page.query_selector_all(selector)
                    for element in elements:
                        data = self.extract_element_data(element, page)
                        if data:
                            self.data.append(data)
                
                # Collect video-specific elements
                video_elements = self.analyze_video_specific_elements(page)
                self.data.extend(video_elements)
                
                # Save collected data
                df = pd.DataFrame(self.data)
                df.to_csv(self.output_dir / 'raw_data.csv', index=False)
                logging.info(f"Collected {len(self.data)} elements")
                
            except Exception as e:
                logging.error(f"Error during data collection: {str(e)}")
            finally:
                browser.close()

if __name__ == "__main__":
    collector = YouTubeDataCollector()
    collector.collect_data()
