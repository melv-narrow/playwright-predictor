"""
Fine-tune the existing element classifier model with Playwright test generation capabilities.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from types import SimpleNamespace
import numpy as np
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA setup and optimization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    device = torch.device('cuda')
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
else:
    device = torch.device('cpu')
    logger.info("CUDA not available, using CPU")

def log_gpu_memory():
    if torch.cuda.is_available():
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

class TestGenerationHead(nn.Module):
    """Additional model head for generating Playwright tests."""
    def __init__(self, input_size: int = 768, hidden_size: int = 512):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.test_generator = nn.Linear(hidden_size, hidden_size)
        self.pattern_classifier = nn.Linear(hidden_size, 7)  # 7 test patterns
        
    def forward(self, features):
        x = self.dropout(self.layer_norm(torch.relu(self.dense(features))))
        test_features = self.test_generator(x)
        patterns = self.pattern_classifier(x)
        return test_features, patterns

class EnhancedElementClassifier(nn.Module):
    """Enhanced model that combines element classification and test generation."""
    def __init__(self, base_model: str, num_patterns: int, num_test_patterns: int, dropout: float = 0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(base_model)
        
        # Move model to CUDA immediately after creation
        if torch.cuda.is_available():
            self.transformer = self.transformer.cuda()
        
        # Only freeze embeddings, keep all encoder layers trainable
        for name, param in self.transformer.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                if torch.cuda.is_available():
                    param.data = param.data.cuda()
        
        # Modified attention layer with increased capacity
        self.attention = nn.MultiheadAttention(768, num_heads=12, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(768)
        self.layer_norm2 = nn.LayerNorm(768)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Projection layers for residual connections with increased width
        self.pattern_proj = nn.Linear(768, 384)
        self.test_proj = nn.Linear(768, 384)
        
        # Task-specific heads with residual connections and deeper architecture
        self.pattern_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pattern_classifier = nn.Linear(384, num_patterns)
        
        self.test_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.test_classifier = nn.Linear(384, num_test_patterns)
        
    def forward(self, input_ids, attention_mask):
        # Get transformer features
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs[0]
        
        # Apply self-attention with residual connection
        attended, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = self.layer_norm1(hidden_states + attended)  # First residual connection
        
        # Pool CLS token and apply dropout
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.layer_norm2(pooled_output)
        
        # Get pattern predictions with residual connections
        pattern_features = self.pattern_head(pooled_output)
        pattern_residual = self.pattern_proj(pooled_output)  # Project to same dimension
        pattern_features = pattern_features + pattern_residual  # Residual connection
        pattern_logits = self.pattern_classifier(pattern_features)
        
        # Get test predictions with residual connections
        test_features = self.test_head(pooled_output)
        test_residual = self.test_proj(pooled_output)  # Project to same dimension
        test_features = test_features + test_residual  # Residual connection
        test_logits = self.test_classifier(test_features)
        
        return SimpleNamespace(
            pattern_logits=pattern_logits,
            test_logits=test_logits
        )

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance and hard examples."""
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class PlaywrightDataset(Dataset):
    """Dataset for Playwright test examples with balanced sampling and augmentation."""
    def __init__(self, examples: List[Dict], tokenizer, augment: bool = True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.augment = augment
        
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
        
        # Calculate class weights for balanced sampling
        pattern_counts = Counter(ex['metadata']['pattern'] for ex in examples)
        total_samples = sum(pattern_counts.values())
        
        # More aggressive weighting for underrepresented classes
        self.class_weights = {
            'assertion': 2.0,     # Boost assertions
            'fixture': 2.0,       # Boost fixtures
            'navigation': 1.5,    # Moderate boost
            'wait_strategy': 1.5, # Moderate boost
            'form_interaction': 1.5,
            'page_object': 1.5,
            'general': 0.3        # Reduce general pattern
        }
        
        self.weights = [
            self.class_weights[ex['metadata']['pattern']]
            for ex in examples
        ]
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def augment_input(self, text: str, pattern: str) -> str:
        """Apply pattern-specific augmentation."""
        if not self.augment:
            return text
            
        if pattern == 'assertion':
            return f"VERIFY: {text} ASSERT THAT: "
        elif pattern == 'fixture':
            return f"SETUP: {text} PREPARE: "
        elif pattern == 'form_interaction':
            return f"INTERACT: {text} FILL FORM: "
        elif pattern == 'navigation':
            return f"NAVIGATE: {text} GO TO: "
        return text
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        pattern = example['metadata']['pattern']
        
        # Add pattern-specific tokens and hints
        prefix = f"[{pattern}] "
        if pattern == 'form_interaction':
            prefix += "FORM: "
        elif pattern == 'navigation':
            prefix += "NAV: "
        elif pattern == 'assertion':
            prefix += "ASSERT: "
        elif pattern == 'wait_strategy':
            prefix += "WAIT: "
        elif pattern == 'fixture':
            prefix += "FIXTURE: "
        
        # Apply augmentation
        augmented_input = self.augment_input(example['input'], pattern)
        
        # Tokenize input with added special tokens
        inputs = self.tokenizer(
            prefix + example['instruction'] + ' ' + augmented_input,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get pattern label
        pattern_idx = self.pattern_map.get(
            pattern,
            self.pattern_map['general']
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pattern': torch.tensor(pattern_idx)
        }

class PlaywrightTrainer:
    def __init__(
        self,
        model: EnhancedElementClassifier,
        tokenizer,
        device: str = 'cuda',
        learning_rate: float = 5e-5,
        weight_decay: float = 0.005
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Use AdamW with weight decay and optimized learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': model.transformer.parameters(), 'lr': learning_rate / 2},
            {'params': model.attention.parameters(), 'lr': learning_rate * 2},
            {'params': model.pattern_head.parameters(), 'lr': learning_rate * 2},
            {'params': model.test_head.parameters(), 'lr': learning_rate * 2}
        ], weight_decay=weight_decay)
        
        # More aggressive class weights
        pattern_weights = torch.tensor([
            2.0,  # assertion (significant boost)
            1.5,  # navigation
            2.0,  # fixture (significant boost)
            1.5,  # wait_strategy
            1.5,  # form_interaction
            1.5,  # page_object
            0.3   # general (significant reduction)
        ]).to(device)
        
        # Use Focal Loss with class weights
        self.pattern_criterion = FocalLoss(
            gamma=2.0,
            weight=pattern_weights
        )
        self.generation_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=0.05
        )
    
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            batch['input_ids'],
            batch['attention_mask']
        )
        
        # Calculate losses
        losses = {
            'pattern': self.pattern_criterion(outputs.pattern_logits, batch['pattern'])
        }
        
        # Combined loss
        total_loss = sum(losses.values())
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        total_loss.backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}

    def validate(self, val_loader):
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch['input_ids'], batch['attention_mask'])
                losses = {
                    'pattern': self.pattern_criterion(outputs.pattern_logits, batch['pattern'])
                }
                val_losses.append({k: v.item() for k, v in losses.items()})
        
        # Calculate average validation loss
        avg_losses = {
            k: sum(d[k] for d in val_losses) / len(val_losses)
            for k in val_losses[0].keys()
        }
        return avg_losses

def finetune(
    base_model_path: str = 'models/best_model.pt',
    playwright_data_dir: str = 'training_data/playwright_docs/training',
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    patience: int = 8,
    warmup_steps: int = 500,
    pin_memory: bool = True
):
    # Log initial GPU state
    log_gpu_memory()
    
    # Load base model
    logger.info("Loading base model...")
    checkpoint = torch.load(base_model_path, map_location=device)
    
    # Default values if config is not found
    num_element_types = checkpoint.get('config', {}).get('num_element_types', 10)
    num_semantic_roles = checkpoint.get('config', {}).get('num_semantic_roles', 7)
    
    logger.info(f"Model has {num_element_types} element types and {num_semantic_roles} semantic roles")
    
    # Create enhanced model
    model = EnhancedElementClassifier(
        base_model='microsoft/codebert-base',
        num_patterns=7,
        num_test_patterns=7
    )
    model = model.to(device)
    
    # Get tokenizer from transformer model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    
    # Load Playwright data
    logger.info("Loading Playwright data...")
    train_data = []
    val_data = []
    
    with open(Path(playwright_data_dir) / 'train.jsonl') as f:
        for line in f:
            train_data.append(json.loads(line))
            
    with open(Path(playwright_data_dir) / 'val.jsonl') as f:
        for line in f:
            val_data.append(json.loads(line))
    
    # Create datasets
    train_dataset = PlaywrightDataset(train_data, tokenizer)
    val_dataset = PlaywrightDataset(val_data, tokenizer)
    
    # Create weighted sampler for balanced training
    sampler = WeightedRandomSampler(
        train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create dataloaders with CUDA optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=pin_memory,  # Enable pin_memory for faster data transfer
        persistent_workers=True  # Keep workers alive between iterations
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    
    # Create trainer
    trainer = PlaywrightTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate
    )
    
    # Create scheduler with warmup
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        trainer.optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop with memory monitoring
    logger.info("Starting fine-tuning...")
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        log_gpu_memory()  # Monitor GPU memory at start of each epoch
        
        # Training
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            losses = trainer.train_step(batch)
            epoch_losses.append(losses)
            scheduler.step()
            
            # Update progress bar
            avg_losses = {
                k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                for k in losses.keys()
            }
            progress_bar.set_postfix(losses=avg_losses)
        
        # Validation
        val_losses = trainer.validate(val_loader)
        val_loss = sum(val_losses.values())
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    # Save enhanced model
    logger.info("Saving enhanced model...")
    output_path = Path('models') / 'playwright_enhanced_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_element_types': num_element_types,
            'num_semantic_roles': num_semantic_roles
        },
        'validation_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }, output_path)
    
    logger.info(f"Model saved to {output_path} with validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    finetune()
