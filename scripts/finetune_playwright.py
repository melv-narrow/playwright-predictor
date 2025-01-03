"""
Fine-tune the existing element classifier model with Playwright test generation capabilities.
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
from model import ElementClassifier
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGenerationHead(nn.Module):
    """Additional model head for generating Playwright tests."""
    def __init__(self, input_size: int = 768, hidden_size: int = 512):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.test_generator = nn.Linear(hidden_size, hidden_size)
        self.pattern_classifier = nn.Linear(hidden_size, 7)  # 7 test patterns
        
    def forward(self, features):
        x = torch.relu(self.dense(features))
        test_features = self.test_generator(x)
        patterns = self.pattern_classifier(x)
        return test_features, patterns

class EnhancedElementClassifier(nn.Module):
    """Enhanced model that combines element classification and test generation."""
    def __init__(self, base_model: ElementClassifier):
        super().__init__()
        self.base_model = base_model
        self.test_head = TestGenerationHead()
        
    def forward(self, input_ids, attention_mask, numerical_features):
        # Get element classification outputs
        element_type_logits, semantic_role_logits = self.base_model(
            input_ids, attention_mask, numerical_features
        )
        
        # Get transformer features for test generation
        transformer_output = self.base_model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = transformer_output[1]  # [CLS] token output
        
        # Generate test features
        test_features, pattern_logits = self.test_head(pooled_output)
        
        return {
            'element_type': element_type_logits,
            'semantic_role': semantic_role_logits,
            'test_features': test_features,
            'pattern': pattern_logits
        }

class PlaywrightDataset(Dataset):
    """Dataset for Playwright test examples."""
    def __init__(self, examples: List[Dict], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        
        # Map patterns to indices
        self.pattern_map = {
            'assertion': 0,
            'navigation': 1,
            'fixture': 2,
            'wait_strategy': 3,
            'form_interaction': 4,
            'page_object': 5,
            'general': 6
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        inputs = self.tokenizer(
            example['instruction'] + ' ' + example['input'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Tokenize output
        outputs = self.tokenizer(
            example['output'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get pattern label
        pattern_idx = self.pattern_map.get(
            example['metadata']['pattern'], 
            self.pattern_map['general']
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'output_ids': outputs['input_ids'].squeeze(),
            'output_mask': outputs['attention_mask'].squeeze(),
            'pattern': torch.tensor(pattern_idx)
        }

class PlaywrightTrainer:
    def __init__(
        self,
        model: EnhancedElementClassifier,
        tokenizer,
        device: str = 'cuda',
        learning_rate: float = 1e-5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.element_criterion = nn.CrossEntropyLoss()
        self.pattern_criterion = nn.CrossEntropyLoss()
        self.generation_criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    
    def train_step(self, batch, element_batch=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            torch.zeros(len(batch['input_ids']), 3).to(self.device)  # Dummy numerical features
        )
        
        # Calculate losses
        losses = {
            'pattern': self.pattern_criterion(outputs['pattern'], batch['pattern'])
        }
        
        # If we have element classification data
        if element_batch is not None:
            element_batch = {k: v.to(self.device) for k, v in element_batch.items()}
            element_outputs = self.model(
                element_batch['input_ids'],
                element_batch['attention_mask'],
                element_batch['numerical_features']
            )
            losses.update({
                'element_type': self.element_criterion(
                    element_outputs['element_type'],
                    element_batch['element_type']
                ),
                'semantic_role': self.element_criterion(
                    element_outputs['semantic_role'],
                    element_batch['semantic_role']
                )
            })
        
        # Combined loss
        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}

def finetune(
    base_model_path: str = 'models/best_model.pt',
    playwright_data_dir: str = 'training_data/playwright_docs/training',
    num_epochs: int = 10,  # Increased to 10
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    device: str = 'cuda'
):
    # Load base model
    logger.info("Loading base model...")
    checkpoint = torch.load(base_model_path, map_location=device)
    
    # Get number of classes from the output layer dimensions
    num_element_types = checkpoint['model_state_dict']['element_type_head.weight'].size(0)
    num_semantic_roles = checkpoint['model_state_dict']['semantic_role_head.weight'].size(0)
    
    logger.info(f"Model has {num_element_types} element types and {num_semantic_roles} semantic roles")
    
    # Create base model
    base_model = ElementClassifier(
        num_element_types=num_element_types,
        num_semantic_roles=num_semantic_roles,
        device=device
    )
    
    # Load state dict
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model = base_model.to(device)
    
    # Get tokenizer from transformer model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    
    # Create enhanced model
    model = EnhancedElementClassifier(base_model)
    model = model.to(device)
    
    # Load Playwright data
    logger.info("Loading Playwright data...")
    train_data = []
    with open(Path(playwright_data_dir) / 'train.jsonl') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    # Create dataset and dataloader
    train_dataset = PlaywrightDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create trainer
    trainer = PlaywrightTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate
    )
    
    # Training loop
    logger.info("Starting fine-tuning...")
    for epoch in range(num_epochs):
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            losses = trainer.train_step(batch)
            epoch_losses.append(losses)
            
            # Update progress bar
            avg_losses = {
                k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                for k in losses.keys()
            }
            progress_bar.set_postfix(losses=avg_losses)
    
    # Save enhanced model
    logger.info("Saving enhanced model...")
    output_path = Path('models') / 'enhanced_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_element_types': num_element_types,
            'num_semantic_roles': num_semantic_roles
        }
    }, output_path)
    
    logger.info(f"Model saved to {output_path}")

if __name__ == '__main__':
    finetune()
