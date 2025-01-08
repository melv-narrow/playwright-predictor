"""
Test GitHub's homepage using the website analyzer.
"""

from website_analyzer import WebsiteAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize analyzer
    analyzer = WebsiteAnalyzer()
    
    # Analyze GitHub homepage
    url = "https://github.com"
    test_patterns = analyzer.generate_tests(url)
    
    # Print analysis results
    logger.info("\nAnalysis Results:")
    for pattern in test_patterns:
        logger.info(f"\nElement Type: {pattern['element_type']}")
        logger.info(f"Pattern: {pattern['pattern']}")
        logger.info(f"Confidence: {pattern['confidence']:.4f}")
        
        if pattern['element_type'] == 'navigation':
            logger.info(f"Text: {pattern['text']}")
            logger.info(f"Link: {pattern['href']}")
        elif pattern['element_type'] == 'form':
            logger.info("Form Inputs:")
            for input_field in pattern['inputs']:
                logger.info(f"  - Type: {input_field['type']}, ID: {input_field['id']}, Name: {input_field['name']}")
        elif pattern['element_type'] == 'header':
            logger.info(f"Text: {pattern['text']}")

if __name__ == '__main__':
    main() 