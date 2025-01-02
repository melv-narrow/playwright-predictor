"""
Element Classifier Model

This module implements the core ML model for classifying HTML elements and determining
their test relevance and appropriate test actions.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from loguru import logger
from typing import Dict, List, Optional, Tuple

class ElementClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_classes: int = 5,
        max_length: int = 512
    ):
        """
        Initialize the element classifier model.
        
        Args:
            model_name: Name of the pretrained transformer model to use
            num_classes: Number of test action classes to predict
            max_length: Maximum sequence length for input tokens
        """
        super().__init__()
        self.max_length = max_length
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {str(e)}")
            raise
            
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(
        self,
        html_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            html_features: Dictionary containing tokenized HTML features
            
        Returns:
            Tensor of class probabilities for each test action
        """
        try:
            outputs = self.transformer(**html_features)
            pooled_output = outputs.pooler_output
            logits = self.classifier(pooled_output)
            return torch.softmax(logits, dim=-1)
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise
            
    def predict(
        self,
        html_elements: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Predict test actions for HTML elements.
        
        Args:
            html_elements: List of HTML element strings to classify
            
        Returns:
            List of (predicted_action, confidence) tuples
        """
        try:
            # Tokenize elements
            features = self.tokenizer(
                html_elements,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Get predictions
            with torch.no_grad():
                predictions = self(features)
            
            # Convert to action labels
            action_map = {
                0: "click",
                1: "input",
                2: "select",
                3: "assert",
                4: "ignore"
            }
            
            results = []
            for pred in predictions:
                action_idx = torch.argmax(pred).item()
                confidence = pred[action_idx].item()
                results.append((action_map[action_idx], confidence))
                
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
            
    def save_model(self, path: str) -> None:
        """Save model weights and tokenizer to disk."""
        try:
            torch.save(self.state_dict(), f"{path}/model.pt")
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
            
    @classmethod
    def load_model(
        cls,
        path: str,
        model_name: str = "microsoft/codebert-base"
    ) -> "ElementClassifier":
        """Load model weights and tokenizer from disk."""
        try:
            model = cls(model_name=model_name)
            model.load_state_dict(torch.load(f"{path}/model.pt"))
            model.tokenizer = AutoTokenizer.from_pretrained(path)
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
