"""
Collect training data from diverse websites with focus on underrepresented elements.
"""

from playwright.sync_api import sync_playwright, Page
import pandas as pd
from pathlib import Path
import time
import logging
from typing import Dict, List, Any
import json

class DiverseSiteCollector:
    def __init__(self):
        self.sites = {
            # Documentation sites (for content and navigation)
            'mdn_docs': {
                'url': 'https://developer.mozilla.org/en-US/docs/Web/HTML/Element',
                'name': 'mdn_docs',
                'selectors': {
                    'content': 'article p, article h1, article h2, article h3, article code, article pre',
                    'navigation': 'nav, .nav-list, .sidebar-inner, .breadcrumbs',
                    'media': 'img, video, audio, svg, canvas',
                }
            },
            'web_dev': {
                'url': 'https://web.dev/learn',
                'name': 'web_dev',
                'selectors': {
                    'content': 'article p, article h1, article h2, article h3, article code, article pre',
                    'navigation': 'nav, .navigation, .sidebar, .breadcrumbs',
                    'media': 'img, video, audio, svg, canvas',
                }
            },
            'css_tricks': {
                'url': 'https://css-tricks.com',
                'name': 'css_tricks',
                'selectors': {
                    'content': 'article p, article h1, article h2, article h3, article code, article pre',
                    'navigation': 'nav, .nav-links, .sidebar, .breadcrumbs',
                    'media': 'img, video, audio, svg, canvas',
                }
            },
            
            # Media-rich sites
            'vimeo': {
                'url': 'https://vimeo.com/watch',
                'name': 'vimeo',
                'selectors': {
                    'media': 'video, img, picture source',
                    'interactive': 'button, a, input, [role="button"]',
                    'navigation': 'nav, [role="navigation"]',
                }
            },
            'dribbble': {
                'url': 'https://dribbble.com/shots',
                'name': 'dribbble',
                'selectors': {
                    'media': 'img, picture source, video',
                    'interactive': 'button, a, input, [role="button"]',
                    'navigation': 'nav, [role="navigation"]',
                }
            },
            'behance': {
                'url': 'https://www.behance.net',
                'name': 'behance',
                'selectors': {
                    'media': 'img, picture source, video',
                    'interactive': 'button, a, input, [role="button"]',
                    'navigation': 'nav, [role="navigation"]',
                }
            },
            
            # Navigation-focused sites
            'amazon': {
                'url': 'https://www.amazon.com',
                'name': 'amazon',
                'selectors': {
                    'navigation': 'nav, [role="navigation"], .nav-menu, .sub-menu, .breadcrumb, header nav, footer nav',
                    'interactive': 'button, a, input, [role="button"]',
                    'content': 'h1, h2, h3, p',
                }
            },
            'reddit': {
                'url': 'https://www.reddit.com',
                'name': 'reddit',
                'selectors': {
                    'navigation': 'nav, [role="navigation"], .top-nav, .side-nav, header nav, footer nav',
                    'interactive': 'button, a, input, [role="button"]',
                    'content': 'h1, h2, h3, p',
                }
            },
            'wikipedia': {
                'url': 'https://www.wikipedia.org',
                'name': 'wikipedia',
                'selectors': {
                    'navigation': 'nav, [role="navigation"], #mw-navigation, .mw-jump-link, .vector-menu-content',
                    'interactive': 'button, a, input, [role="button"]',
                    'content': 'h1, h2, h3, p, .mw-parser-output p',
                }
            },
            
            # Documentation with interactive examples
            'tailwind': {
                'url': 'https://tailwindcss.com/docs',
                'name': 'tailwind',
                'selectors': {
                    'content': 'article p, article h1, article h2, article h3, article code, article pre',
                    'navigation': 'nav, .navigation, .sidebar',
                    'interactive': 'button, a, input, [role="button"]',
                }
            },
            'vuejs': {
                'url': 'https://vuejs.org/guide/introduction.html',
                'name': 'vuejs',
                'selectors': {
                    'content': 'article p, article h1, article h2, article h3, article code, article pre',
                    'navigation': 'nav, .navigation, .sidebar',
                    'interactive': 'button, a, input, [role="button"]',
                }
            },
            
            # Testing-focused sites
            'selenium_dev': {
                'url': 'https://www.selenium.dev/documentation/',
                'name': 'selenium_dev',
                'selectors': {
                    'content': 'article p, article h1, article h2, article h3, article code, article pre',
                    'navigation': 'nav, .navigation, .sidebar',
                    'interactive': 'button, a, input, [role="button"]',
                }
            },
            'cypress': {
                'url': 'https://docs.cypress.io',
                'name': 'cypress',
                'selectors': {
                    'content': 'article p, article h1, article h2, article h3, article code, article pre',
                    'navigation': 'nav, .navigation, .sidebar',
                    'interactive': 'button, a, input, [role="button"]',
                }
            }
        }
        
        # Additional selectors for better coverage
        self.common_selectors = {
            'navigation': '[role="navigation"], header nav, footer nav, .main-nav, .sub-nav, .menu-nav, .breadcrumbs, .pagination, .tabs, [role="menubar"], [role="tablist"]',
            'content': 'article, section p, main p, .content p, .article-content, blockquote, dl, dt, dd, ol li, ul li',
            'media': '[role="img"], figure img, picture img, video, audio, canvas, svg, [role="figure"], [role="img"]',
            'interactive': '[role="button"], [role="link"], [role="tab"], [role="menuitem"], [role="checkbox"], [role="radio"], [role="combobox"], [role="textbox"], [role="searchbox"]'
        }
        
        self.output_dir = Path('training_data/diverse_sites')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self.setup_logging()

    def extract_element_data(self, element: Any, page: Page, category: str) -> Dict:
        """Extract relevant data from a DOM element."""
        try:
            box = element.bounding_box()
            attrs = element.evaluate('el => Object.assign({}, ...Array.from(el.attributes, ({name, value}) => ({[name]: value})))')
            
            return {
                'url': page.url,
                'site_category': category,
                'tag_name': element.evaluate('el => el.tagName.toLowerCase()'),
                'attributes': attrs,
                'inner_text': element.inner_text().strip(),
                'is_visible': element.is_visible(),
                'bounding_box': box,
                'selector': element.evaluate('el => cssPath(el)'),
                'aria_role': element.get_attribute('role'),
                'aria_label': element.get_attribute('aria-label'),
                'classes': element.get_attribute('class'),
                'id': element.get_attribute('id'),
                'is_clickable': element.is_enabled() and element.is_visible(),
                'has_hover': bool(element.get_attribute('onmouseover') or element.get_attribute('onmouseenter')),
                'text_length': len(element.inner_text().strip()),
                'child_elements': element.evaluate('el => el.children.length'),
                'parent_tag': element.evaluate('el => el.parentElement ? el.parentElement.tagName.toLowerCase() : null')
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

    def handle_cookie_banners(self, page: Page):
        """Handle common cookie consent banners."""
        common_selectors = [
            # Common cookie banner accept buttons
            '[id*="cookie"] button', '[class*="cookie"] button',
            '[id*="consent"] button', '[class*="consent"] button',
            'button[class*="accept"]', 'button[class*="agree"]',
            # Specific selectors for common sites
            '.fc-button-label', '.cookie-notice-accept',
            '#onetrust-accept-btn-handler',
            '[aria-label*="Accept"]', '[aria-label*="agree"]'
        ]
        
        for selector in common_selectors:
            try:
                accept_button = page.query_selector(selector)
                if accept_button and accept_button.is_visible():
                    accept_button.click()
                    logging.info(f"Clicked cookie banner using selector: {selector}")
                    page.wait_for_timeout(1000)  # Wait for banner to disappear
                    break
            except Exception as e:
                logging.debug(f"Failed to handle cookie banner with selector {selector}: {str(e)}")

    def scroll_page(self, page: Page):
        """Scroll the page to load lazy content."""
        try:
            # Get initial page height
            initial_height = page.evaluate('document.body.scrollHeight')
            
            # Scroll in increments
            current_position = 0
            scroll_increment = 300
            
            while current_position < initial_height:
                page.evaluate(f'window.scrollTo(0, {current_position})')
                page.wait_for_timeout(100)  # Wait for content to load
                current_position += scroll_increment
                
                # Check if new content was loaded
                new_height = page.evaluate('document.body.scrollHeight')
                if new_height > initial_height:
                    initial_height = new_height
                    
                # Prevent infinite loops
                if current_position > 10000:  # Max scroll limit
                    break
                    
            # Scroll back to top
            page.evaluate('window.scrollTo(0, 0)')
            
        except Exception as e:
            logging.error(f"Error during page scroll: {str(e)}")

    def collect_site_data(self, site: Dict[str, str], page: Page):
        """Collect data from a specific site."""
        try:
            logging.info(f"Collecting data from {site['url']}")
            page.goto(site['url'], wait_until='networkidle')
            
            # Handle cookie consent if present
            self.handle_cookie_banners(page)
            
            # Scroll to load lazy content
            self.scroll_page(page)
            
            # Wait for dynamic content
            page.wait_for_timeout(2000)
            
            # Inject helper functions
            self.inject_helper_functions(page)
            
            site_data = []
            for category, selectors in site['selectors'].items():
                for selector in selectors.split(','):
                    try:
                        elements = page.query_selector_all(selector.strip())
                        for element in elements:
                            data = self.extract_element_data(element, page, category)
                            if data:
                                site_data.append(data)
                    except Exception as e:
                        logging.error(f"Error collecting elements with selector {selector}: {str(e)}")
            
            # Add common selectors
            for category, selectors in self.common_selectors.items():
                for selector in selectors.split(','):
                    try:
                        elements = page.query_selector_all(selector.strip())
                        for element in elements:
                            data = self.extract_element_data(element, page, category)
                            if data:
                                site_data.append(data)
                    except Exception as e:
                        logging.error(f"Error collecting elements with selector {selector}: {str(e)}")
            
            # Save site data
            output_file = self.output_dir / f"{site['name']}_raw.csv"
            df = pd.DataFrame(site_data)
            df.to_csv(output_file, index=False)
            logging.info(f"Saved {len(site_data)} elements from {site['url']}")
            
            return len(site_data)
        except Exception as e:
            logging.error(f"Error collecting data from {site['url']}: {str(e)}")
            return 0

    def collect_all_sites(self):
        """Collect data from all configured sites."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            
            total_elements = 0
            for site in self.sites.values():
                elements_count = self.collect_site_data(site, page)
                total_elements += elements_count
            
            browser.close()
            logging.info(f"\nCollection complete. Total elements: {total_elements}")

    def setup_logging(self):
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'collector.log'),
                logging.StreamHandler()
            ]
        )

def main():
    collector = DiverseSiteCollector()
    collector.collect_all_sites()

if __name__ == "__main__":
    main()
