"""
Configuration Manager

Handles loading and validation of configuration settings for the AI test generator.
"""

import os
import yaml
from typing import Any, Dict, Optional
from loguru import logger
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    name: str
    max_length: int
    num_classes: int
    confidence_threshold: float

@dataclass
class ScraperConfig:
    headless: bool
    viewport: Dict[str, int]
    timeout: int
    wait_for_load: bool
    max_depth: int
    excluded_patterns: list[str]

@dataclass
class TestConfig:
    framework: str
    language: str
    test_template: str
    assertion_confidence: float
    max_actions_per_test: int
    generate_comments: bool
    retry_attempts: int
    timeout: int

@dataclass
class OutputConfig:
    base_dir: str
    screenshots_dir: str
    reports_dir: str
    log_level: str

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to custom config file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "default_config.yaml"
        )
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration from file."""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_path}"
                )
                
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            self._validate_config(config)
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
            
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure and values."""
        required_sections = ['model', 'scraper', 'test_generation', 'output']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
                
        # Validate model config
        model_config = config['model']
        if not isinstance(model_config.get('confidence_threshold', 0), (int, float)):
            raise ValueError("Model confidence threshold must be a number")
            
        # Validate test generation config
        test_config = config['test_generation']
        if test_config.get('language') not in ['typescript', 'javascript']:
            raise ValueError("Test language must be typescript or javascript")
            
    @property
    def model_config(self) -> ModelConfig:
        """Get model configuration settings."""
        return ModelConfig(**self.config['model'])
        
    @property
    def scraper_config(self) -> ScraperConfig:
        """Get scraper configuration settings."""
        return ScraperConfig(**self.config['scraper'])
        
    @property
    def test_config(self) -> TestConfig:
        """Get test generation configuration settings."""
        return TestConfig(**self.config['test_generation'])
        
    @property
    def output_config(self) -> OutputConfig:
        """Get output configuration settings."""
        return OutputConfig(**self.config['output'])
        
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific configuration value.
        
        Args:
            section: Configuration section (e.g., 'model', 'scraper')
            key: Configuration key to update
            value: New value to set
        """
        if section not in self.config:
            raise ValueError(f"Invalid config section: {section}")
            
        if key not in self.config[section]:
            raise ValueError(f"Invalid config key: {key}")
            
        self.config[section][key] = value
        
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Optional custom save path
        """
        try:
            save_path = output_path or self.config_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise
