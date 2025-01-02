"""
AI Test Generator Main Entry Point

This module provides the main CLI interface for the AI test generator framework.
"""

import asyncio
import click
from loguru import logger
import os
from typing import Optional

from utils.config_manager import ConfigManager
from test_generator.generator import TestGenerator
from scraper.webpage_analyzer import WebpageAnalyzer
from ai.models.element_classifier import ElementClassifier

@click.group()
def cli():
    """AI-powered test automation framework for Playwright."""
    pass

@cli.command()
@click.argument('url')
@click.option('--config', '-c', help='Path to custom config file')
@click.option('--output', '-o', help='Output directory for generated tests')
@click.option('--headless/--no-headless', default=True, help='Run in headless mode')
def generate(url: str, config: Optional[str], output: Optional[str], headless: bool):
    """Generate Playwright tests for a webpage."""
    try:
        # Initialize components
        config_manager = ConfigManager(config) if config else ConfigManager()
        if headless is not None:
            config_manager.update_config('scraper', 'headless', headless)
            
        generator = TestGenerator(config_manager)
        
        # Generate tests
        output_path = asyncio.run(generator.generate_tests(url, output))
        click.echo(f"Successfully generated tests: {output_path}")
        
    except Exception as e:
        logger.error(f"Test generation failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('url')
@click.option('--output', '-o', help='Output directory for analysis data')
def analyze(url: str, output: Optional[str]):
    """Analyze a webpage and save element data."""
    try:
        analyzer = WebpageAnalyzer()
        output_dir = output or "./analysis_data"
        
        asyncio.run(analyzer.save_analysis(url, output_dir))
        click.echo(f"Analysis data saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Page analysis failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('model_path')
@click.argument('data_dir')
@click.option('--epochs', default=10, help='Number of training epochs')
def train(model_path: str, data_dir: str, epochs: int):
    """Train the element classifier model."""
    try:
        # Initialize model
        model = ElementClassifier()
        
        # Load training data
        if not os.path.exists(data_dir):
            raise click.ClickException(f"Training data directory not found: {data_dir}")
            
        # Train model (placeholder for actual training logic)
        click.echo("Training model... (not implemented)")
        
        # Save trained model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        click.echo(f"Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli()
