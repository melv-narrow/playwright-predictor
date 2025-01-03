"""
Script to download Playwright documentation and convert to markdown.
"""

import os
import logging
from urllib.parse import urljoin
import asyncio
from playwright.async_api import async_playwright
import html2text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlaywrightDocsDownloader:
    def __init__(self):
        self.base_url = "https://playwright.dev/docs/intro"
        self.docs_base = "https://playwright.dev"
        self.output_dir = os.path.join("training_data", "playwright_docs", "markdown")
        
        # Configure HTML to Markdown converter
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False  # Keep links for reference
        self.html2text.ignore_images = True
        self.html2text.ignore_emphasis = False
        self.html2text.body_width = 0  # Don't wrap lines
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    async def collect_doc_links(self, page) -> list[str]:
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

    def get_filename_from_url(self, url: str) -> str:
        """Convert URL to a valid filename."""
        # Remove base URL and leading slash
        filename = url.replace(self.docs_base, '').replace('/docs/', '')
        # Replace remaining slashes with underscores
        filename = filename.replace('/', '_')
        return f"{filename}.md"

    async def download_page(self, page, url: str) -> None:
        """Download a single documentation page and convert to markdown."""
        try:
            # Navigate and wait for content
            await page.goto(url, wait_until='networkidle')
            await page.wait_for_selector('article', state='attached')
            
            # Get the HTML content
            html_content = await page.evaluate('() => document.querySelector("article").outerHTML')
            
            # Convert HTML to Markdown
            markdown = self.html2text.handle(html_content)
            
            # Save to file
            filename = self.get_filename_from_url(url)
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # Add source URL as reference
                f.write(f"Source: {url}\n\n")
                f.write(markdown)
            
            logger.info(f"Saved {url} to {filename}")
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")

    async def download_all_docs(self):
        """Download all Playwright documentation pages."""
        logger.info("Starting documentation download...")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # Collect all documentation links
            await page.goto(self.base_url)
            doc_links = await self.collect_doc_links(page)
            
            # Download each page
            for url in doc_links:
                logger.info(f"Processing {url}")
                await self.download_page(page, url)
            
            await browser.close()
            
        logger.info("Documentation download complete!")

async def main():
    downloader = PlaywrightDocsDownloader()
    await downloader.download_all_docs()

if __name__ == "__main__":
    asyncio.run(main())
