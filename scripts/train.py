"""
Training script for the element classifier model.
"""

import logging
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.cuda.amp as amp

from prepare_training_data import DataPreparator
from model import ElementClassifier, ElementTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train(
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    data_dir: str = 'training_data/normalized',
    output_dir: str = 'models'
):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Enable cuDNN autotuner if using CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Prepare data
    logging.info('Preparing training data...')
    preparator = DataPreparator(data_dir)
    
    # First prepare features and labels
    df = preparator.load_normalized_data()
    features, labels = preparator.prepare_features(df)
    
    # Then create data loaders
    train_loader, val_loader, test_loader = preparator.prepare_data_loaders(
        features=features,
        labels=labels,
        batch_size=batch_size
    )
    
    # Initialize model
    model = ElementClassifier(
        num_element_types=len(preparator.element_type_encoder.classes_),
        num_semantic_roles=len(preparator.semantic_role_encoder.classes_),
        device=device
    ).to(device)
    
    # Initialize trainer with mixed precision
    trainer = ElementTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    logging.info('Starting training...')
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            # Training step
            with torch.cuda.amp.autocast():
                losses = trainer.train_step(batch)
                
            # Backward pass with gradient scaling
            scaler.scale(losses['total']).backward()
            scaler.step(trainer.optimizer)
            scaler.update()
            trainer.optimizer.zero_grad(set_to_none=True)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'elem_loss': f"{losses['element_type'].item():.4f}",
                'role_loss': f"{losses['semantic_role'].item():.4f}"
            })
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                with torch.cuda.amp.autocast():
                    losses = trainer.validation_step(batch)
                    val_losses.append(losses['total'].item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        logging.info(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'element_types': preparator.element_type_encoder.classes_.tolist(),
            'semantic_roles': preparator.semantic_role_encoder.classes_.tolist()
        }
        
        torch.save(checkpoint, Path(output_dir) / f'checkpoint_epoch_{epoch + 1}.pt')
        logging.info(f'Saved checkpoint for epoch {epoch + 1}')

if __name__ == '__main__':
    train()
