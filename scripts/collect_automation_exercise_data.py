"""
Data Collection Script for AutomationExercise.com

This script collects training data from the Automation Exercise website,
focusing on common test scenarios and interactive elements.
"""

import asyncio
import os
from pathlib import Path
from loguru import logger
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collection.collector import DataCollector

# Define important pages to analyze
TARGET_PAGES = [
    "https://www.automationexercise.com/",
    "https://www.automationexercise.com/login",
    "https://www.automationexercise.com/products",
    "https://www.automationexercise.com/contact_us",
    "https://www.automationexercise.com/test_cases"
]

async def main():
    # Initialize collector
    output_dir = os.path.join(project_root, "training_data", "automation_exercise")
    collector = DataCollector(
        output_dir=output_dir,
        headless=False,  # Set to False to see the browser during collection
        viewport_size={"width": 1280, "height": 800}
    )
    
    try:
        # Collect data from each target page
        for url in TARGET_PAGES:
            logger.info(f"Collecting data from {url}")
            dataset_path = await collector.collect_from_url(
                url=url,
                max_depth=1,  # Stay on the same page initially
                max_pages=1    # Don't follow links yet
            )
            logger.info(f"Data saved to {dataset_path}")
            
    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        raise
        
    logger.info("Data collection completed!")

if __name__ == "__main__":
    # Set up logging
    logger.add(
        os.path.join(project_root, "logs", "data_collection.log"),
        rotation="1 day"
    )
    
    # Run the collector
    asyncio.run(main())
