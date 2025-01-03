"""
Test the trained model on specific examples.
"""

import torch
import json
import logging
from pathlib import Path
from prepare_training_data import DataPreparator
from model import ElementClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def predict_element(
    model,
    preparator,
    tag: str,
    inner_text: str,
    selector: str,
    attributes: str,
    is_interactive: bool,
    is_visible: bool,
    has_text: bool,
    device: str
):
    """Make predictions for a single element."""
    # Prepare text input
    text = f"""
    Tag: {tag}
    Text: {inner_text}
    Selector: {selector}
    Attributes: {attributes}
    """
    
    # Tokenize
    encodings = preparator.tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Prepare numerical features
    numerical_features = torch.tensor([[
        float(is_interactive),
        float(is_visible),
        float(has_text)
    ]], dtype=torch.float32)
    
    # Move to device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    numerical_features = numerical_features.to(device)
    
    # Get predictions
    with torch.no_grad():
        element_logits, role_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numerical_features=numerical_features
        )
    
    # Get predicted classes
    element_pred = torch.argmax(element_logits, dim=1).cpu().item()
    role_pred = torch.argmax(role_logits, dim=1).cpu().item()
    
    # Get class names
    element_type = preparator.element_type_encoder.classes_[element_pred]
    semantic_role = preparator.semantic_role_encoder.classes_[role_pred]
    
    # Get probabilities
    element_probs = torch.softmax(element_logits, dim=1).cpu().numpy()[0]
    role_probs = torch.softmax(role_logits, dim=1).cpu().numpy()[0]
    
    return {
        'element_type': element_type,
        'semantic_role': semantic_role,
        'element_confidence': float(element_probs[element_pred]),
        'role_confidence': float(role_probs[role_pred])
    }

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load checkpoint
    checkpoint_path = 'models/checkpoint_epoch_10.pt'
    logging.info(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Load label encoders
    with open('training_data/processed/element_type_encoder.json', 'r') as f:
        element_type_data = json.load(f)
    with open('training_data/processed/semantic_role_encoder.json', 'r') as f:
        semantic_role_data = json.load(f)
    
    # Initialize data preparator and model
    preparator = DataPreparator()
    preparator.element_type_encoder.classes_ = element_type_data['classes_']
    preparator.semantic_role_encoder.classes_ = semantic_role_data['classes_']
    
    model = ElementClassifier(
        num_element_types=len(preparator.element_type_encoder.classes_),
        num_semantic_roles=len(preparator.semantic_role_encoder.classes_),
        device=device
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test examples
    test_cases = [
        {
            "tag": "button",
            "inner_text": "Login",
            "selector": "#login-button",
            "attributes": "class='btn btn-primary'",
            "is_interactive": True,
            "is_visible": True,
            "has_text": True
        },
        {
            "tag": "input",
            "inner_text": "",
            "selector": "#username",
            "attributes": "type='text' placeholder='Enter username'",
            "is_interactive": True,
            "is_visible": True,
            "has_text": False
        },
        {
            "tag": "h1",
            "inner_text": "Welcome to our website",
            "selector": ".header-title",
            "attributes": "class='title'",
            "is_interactive": False,
            "is_visible": True,
            "has_text": True
        },
        {
            "tag": "a",
            "inner_text": "Read more",
            "selector": ".article-link",
            "attributes": "href='/article/123'",
            "is_interactive": True,
            "is_visible": True,
            "has_text": True
        },
        {
            "tag": "img",
            "inner_text": "",
            "selector": ".product-image",
            "attributes": "src='/images/product.jpg' alt='Product photo'",
            "is_interactive": False,
            "is_visible": True,
            "has_text": False
        }
    ]
    
    # Make predictions
    logging.info("\nTesting examples:")
    for i, test_case in enumerate(test_cases, 1):
        logging.info(f"\nExample {i}:")
        logging.info(f"Input:")
        logging.info(f"  Tag: {test_case['tag']}")
        logging.info(f"  Text: {test_case['inner_text']}")
        logging.info(f"  Selector: {test_case['selector']}")
        logging.info(f"  Attributes: {test_case['attributes']}")
        
        result = predict_element(
            model,
            preparator,
            **test_case,
            device=device
        )
        
        logging.info("Predictions:")
        logging.info(f"  Element Type: {result['element_type']} (confidence: {result['element_confidence']:.2%})")
        logging.info(f"  Semantic Role: {result['semantic_role']} (confidence: {result['role_confidence']:.2%})")

if __name__ == '__main__':
    main()
