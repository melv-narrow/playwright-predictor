"""
Define the model architecture for element classification.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple

class ElementClassifier(nn.Module):
    def __init__(
        self,
        num_element_types: int,
        num_semantic_roles: int,
        numerical_features: int = 3,
        hidden_size: int = 768,
        dropout: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained('microsoft/codebert-base').to(device)
        
        # Freeze transformer parameters (optional)
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Numerical features processing
        self.numerical_processor = nn.Sequential(
            nn.Linear(numerical_features, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(device)
        
        # Combined features processing
        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(device)
        
        # Task-specific heads
        self.element_type_head = nn.Linear(hidden_size // 2, num_element_types).to(device)
        self.semantic_role_head = nn.Linear(hidden_size // 2, num_semantic_roles).to(device)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure inputs are on correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        numerical_features = numerical_features.to(self.device)
        
        # Process text through transformer
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get CLS token representation
        cls_output = transformer_output.last_hidden_state[:, 0]
        
        # Process numerical features
        numerical_output = self.numerical_processor(numerical_features)
        
        # Combine features
        combined = torch.cat([cls_output, numerical_output], dim=1)
        shared_features = self.combined_processor(combined)
        
        # Task-specific predictions
        element_type_logits = self.element_type_head(shared_features)
        semantic_role_logits = self.semantic_role_head(shared_features)
        
        return element_type_logits, semantic_role_logits

class ElementTrainer:
    def __init__(
        self,
        model: ElementClassifier,
        device: str = 'cuda',
        learning_rate: float = 2e-5,
        element_type_weight: float = 1.0,
        semantic_role_weight: float = 1.0
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.element_type_weight = element_type_weight
        self.semantic_role_weight = semantic_role_weight
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single training step."""
        self.optimizer.zero_grad()
        
        # Unpack batch
        features, labels = batch
        
        # Move batch to device
        numerical_features = features['numerical_features'].to(self.device)
        input_ids = features['input_ids'].to(self.device)
        attention_mask = features['attention_mask'].to(self.device)
        
        # Split labels for element type and semantic role
        element_type_labels = labels[:, 0].to(self.device)
        semantic_role_labels = labels[:, 1].to(self.device)
        
        # Forward pass
        element_type_logits, semantic_role_logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numerical_features=numerical_features
        )
        
        # Calculate losses
        element_type_loss = self.criterion(element_type_logits, element_type_labels)
        semantic_role_loss = self.criterion(semantic_role_logits, semantic_role_labels)
        
        # Combine losses
        total_loss = (
            self.element_type_weight * element_type_loss +
            self.semantic_role_weight * semantic_role_loss
        )
        
        return {
            'total': total_loss,
            'element_type': element_type_loss,
            'semantic_role': semantic_role_loss,
            'element_logits': element_type_logits,
            'role_logits': semantic_role_logits
        }
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single validation step."""
        self.model.eval()
        
        # Unpack batch
        features, labels = batch
        
        # Move batch to device
        numerical_features = features['numerical_features'].to(self.device)
        input_ids = features['input_ids'].to(self.device)
        attention_mask = features['attention_mask'].to(self.device)
        
        # Split labels for element type and semantic role
        element_type_labels = labels[:, 0].to(self.device)
        semantic_role_labels = labels[:, 1].to(self.device)
        
        with torch.no_grad():
            # Forward pass
            element_type_logits, semantic_role_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features
            )
            
            # Calculate losses
            element_type_loss = self.criterion(element_type_logits, element_type_labels)
            semantic_role_loss = self.criterion(semantic_role_logits, semantic_role_labels)
            
            # Combine losses
            total_loss = (
                self.element_type_weight * element_type_loss +
                self.semantic_role_weight * semantic_role_loss
            )
        
        return {
            'total': total_loss,
            'element_type': element_type_loss,
            'semantic_role': semantic_role_loss,
            'element_logits': element_type_logits,
            'role_logits': semantic_role_logits
        }

class ElementClassifierTrainer:
    def __init__(
        self,
        model: ElementClassifier,
        device: torch.device,
        learning_rate: float = 2e-5,
        element_type_weight: float = 1.0,
        semantic_role_weight: float = 1.0
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.element_type_weight = element_type_weight
        self.semantic_role_weight = semantic_role_weight
    
    def train_step(self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        features, labels = batch
        
        # Move data to device
        features = {k: v.to(self.device) for k, v in features.items()}
        labels = labels.to(self.device)
        
        # Forward pass
        element_type_logits, semantic_role_logits = self.model(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
            numerical_features=features['numerical_features']
        )
        
        # Calculate losses
        element_type_loss = self.criterion(element_type_logits, labels[:, 0])
        semantic_role_loss = self.criterion(semantic_role_logits, labels[:, 1])
        
        # Combined loss
        loss = (
            self.element_type_weight * element_type_loss + 
            self.semantic_role_weight * semantic_role_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'element_type_loss': element_type_loss.item(),
            'semantic_role_loss': semantic_role_loss.item()
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model on dataloader."""
        self.model.eval()
        total_element_type_loss = 0
        total_semantic_role_loss = 0
        total_element_type_correct = 0
        total_semantic_role_correct = 0
        total_samples = 0
        
        for features, labels in dataloader:
            # Move data to device
            features = {k: v.to(self.device) for k, v in features.items()}
            labels = labels.to(self.device)
            
            # Forward pass
            element_type_logits, semantic_role_logits = self.model(
                input_ids=features['input_ids'],
                attention_mask=features['attention_mask'],
                numerical_features=features['numerical_features']
            )
            
            # Calculate losses
            element_type_loss = self.criterion(element_type_logits, labels[:, 0])
            semantic_role_loss = self.criterion(semantic_role_logits, labels[:, 1])
            
            # Calculate accuracy
            element_type_pred = element_type_logits.argmax(dim=1)
            semantic_role_pred = semantic_role_logits.argmax(dim=1)
            
            total_element_type_correct += (element_type_pred == labels[:, 0]).sum().item()
            total_semantic_role_correct += (semantic_role_pred == labels[:, 1]).sum().item()
            
            total_element_type_loss += element_type_loss.item() * len(labels)
            total_semantic_role_loss += semantic_role_loss.item() * len(labels)
            total_samples += len(labels)
        
        return {
            'element_type_loss': total_element_type_loss / total_samples,
            'semantic_role_loss': total_semantic_role_loss / total_samples,
            'element_type_accuracy': total_element_type_correct / total_samples,
            'semantic_role_accuracy': total_semantic_role_correct / total_samples
        }
