#!/usr/bin/env python
"""
Test Generation Script

This script generates Playwright test scripts for web pages using
our trained ML model to predict appropriate test actions.
"""

import asyncio
import sys
from pathlib import Path
import argparse
from loguru import logger

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.test_generation.generator import TestGenerator

async def main(args):
    """Main execution function."""
    try:
        # Initialize generator
        generator = TestGenerator(
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            confidence_threshold=args.confidence
        )
        
        # Generate test suite
        test_file = await generator.generate_test_suite(
            url=args.url,
            test_name=args.name,
            description=args.description
        )
        
        if test_file:
            logger.info(f"Successfully generated test: {test_file}")
        else:
            logger.error("Failed to generate test suite")
            
    except Exception as e:
        logger.error(f"Test generation failed: {str(e)}")
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Playwright tests using ML predictions"
    )
    
    parser.add_argument(
        "--url",
        required=True,
        help="URL of the page to test"
    )
    
    parser.add_argument(
        "--name",
        required=True,
        help="Name for the generated test"
    )
    
    parser.add_argument(
        "--description",
        help="Description of the test"
    )
    
    parser.add_argument(
        "--output-dir",
        default="generated_tests",
        help="Directory to save generated tests"
    )
    
    parser.add_argument(
        "--model-dir",
        default="models/rf",
        help="Directory containing trained model"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="Minimum confidence threshold for predictions"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_path = Path(args.output_dir) / "generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, rotation="1 day", retention="7 days", level="INFO")
    
    # Run generator
    asyncio.run(main(args))
