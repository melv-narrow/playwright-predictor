"""
Evaluate the trained element classifier model.
"""

import logging
import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from prepare_training_data import DataPreparator
from model import ElementClassifier, ElementTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def plot_confusion_matrix(cm, classes, title, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate(
    checkpoint_path: str = 'models/checkpoint_epoch_10.pt',
    data_dir: str = 'training_data/normalized',
    output_dir: str = 'evaluation_results'
):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logging.info(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)

    # Prepare data
    logging.info('Preparing test data...')
    preparator = DataPreparator(data_dir)
    df = preparator.load_normalized_data()
    features, labels = preparator.prepare_features(df)
    _, _, test_loader = preparator.prepare_data_loaders(features, labels)

    # Initialize model
    model = ElementClassifier(
        num_element_types=len(preparator.element_type_encoder.classes_),
        num_semantic_roles=len(preparator.semantic_role_encoder.classes_),
        device=device
    ).to(device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize metrics
    all_element_preds = []
    all_element_labels = []
    all_role_preds = []
    all_role_labels = []
    test_losses = []

    # Initialize trainer for loss calculation
    trainer = ElementTrainer(model=model, device=device)

    # Evaluate
    logging.info('Starting evaluation...')
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            # Get predictions
            features, labels = batch
            
            # Move batch to device
            input_ids = features['input_ids'].to(device)
            attention_mask = features['attention_mask'].to(device)
            numerical_features = features['numerical_features'].to(device)
            
            # Forward pass
            element_logits, role_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features
            )

            # Get predictions
            element_preds = torch.argmax(element_logits, dim=1).cpu().numpy()
            role_preds = torch.argmax(role_logits, dim=1).cpu().numpy()
            
            # Get labels
            element_labels = labels[:, 0].cpu().numpy()
            role_labels = labels[:, 1].cpu().numpy()

            # Store predictions and labels
            all_element_preds.extend(element_preds)
            all_element_labels.extend(element_labels)
            all_role_preds.extend(role_preds)
            all_role_labels.extend(role_labels)

            # Calculate loss
            losses = trainer.validation_step(batch)
            test_losses.append(losses['total'].item())

    # Calculate metrics
    element_types = preparator.element_type_encoder.classes_
    semantic_roles = preparator.semantic_role_encoder.classes_

    # Element type metrics
    element_report = classification_report(
        all_element_labels, 
        all_element_preds, 
        target_names=element_types,
        output_dict=True
    )
    element_cm = confusion_matrix(all_element_labels, all_element_preds)

    # Semantic role metrics
    role_report = classification_report(
        all_role_labels, 
        all_role_preds, 
        target_names=semantic_roles,
        output_dict=True
    )
    role_cm = confusion_matrix(all_role_labels, all_role_preds)

    # Plot confusion matrices
    plot_confusion_matrix(
        element_cm, 
        element_types, 
        'Element Type Confusion Matrix',
        output_dir / 'element_type_cm.png'
    )
    plot_confusion_matrix(
        role_cm, 
        semantic_roles, 
        'Semantic Role Confusion Matrix',
        output_dir / 'semantic_role_cm.png'
    )

    # Save metrics
    metrics = {
        'test_loss': np.mean(test_losses),
        'element_type_metrics': element_report,
        'semantic_role_metrics': role_report
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Log results
    logging.info(f"Test Loss: {metrics['test_loss']:.4f}")
    logging.info("\nElement Type Classification Report:")
    logging.info(f"Accuracy: {element_report['accuracy']:.4f}")
    logging.info(f"Macro F1: {element_report['macro avg']['f1-score']:.4f}")
    logging.info(f"Weighted F1: {element_report['weighted avg']['f1-score']:.4f}")
    
    logging.info("\nSemantic Role Classification Report:")
    logging.info(f"Accuracy: {role_report['accuracy']:.4f}")
    logging.info(f"Macro F1: {role_report['macro avg']['f1-score']:.4f}")
    logging.info(f"Weighted F1: {role_report['weighted avg']['f1-score']:.4f}")

def evaluate_model(model_path):
    # Load the model
    model = ElementClassifier(
        num_element_types=5,
        num_semantic_roles=6
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(model)

if __name__ == '__main__':
    model_path = 'models/enhanced_model.pt'
    evaluate_model(model_path)
    evaluate()
