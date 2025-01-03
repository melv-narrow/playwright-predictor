"""
Analyze elements from a live website using Playwright and our trained model.
"""

import asyncio
import torch
import json
import logging
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from prepare_training_data import DataPreparator
from model import ElementClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def extract_element_info(element):
    """Extract relevant information from a Playwright element."""
    tag = await element.evaluate('el => el.tagName.toLowerCase()')
    
    # Get inner text
    try:
        inner_text = await element.inner_text()
    except:
        inner_text = ""
    
    # Get selector
    try:
        selector = await element.evaluate('el => el.id ? "#" + el.id : el.className ? "." + el.className.split(" ")[0] : el.tagName.toLowerCase()')
    except:
        selector = tag
    
    # Get attributes
    try:
        attrs = await element.evaluate('''el => {
            const attrs = {};
            for (const attr of el.attributes) {
                attrs[attr.name] = attr.value;
            }
            return attrs;
        }''')
        attributes = ' '.join([f'{k}="{v}"' for k, v in attrs.items()])
    except:
        attributes = ""
    
    # Check visibility and interactivity
    try:
        is_visible = await element.is_visible()
    except:
        is_visible = False
    
    try:
        is_interactive = await element.evaluate('''el => {
            const interactive_tags = ['a', 'button', 'input', 'select', 'textarea'];
            return interactive_tags.includes(el.tagName.toLowerCase()) || 
                   el.onclick != null || 
                   el.getAttribute('role') === 'button';
        }''')
    except:
        is_interactive = False
    
    has_text = bool(inner_text.strip())
    
    return {
        'tag': tag,
        'inner_text': inner_text,
        'selector': selector,
        'attributes': attributes,
        'is_interactive': is_interactive,
        'is_visible': is_visible,
        'has_text': has_text
    }

def predict_element(model, preparator, element_info, device):
    """Make predictions for a single element."""
    # Prepare text input
    text = f"""
    Tag: {element_info['tag']}
    Text: {element_info['inner_text']}
    Selector: {element_info['selector']}
    Attributes: {element_info['attributes']}
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
        float(element_info['is_interactive']),
        float(element_info['is_visible']),
        float(element_info['has_text'])
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

async def analyze_website(url: str, model, preparator, device):
    """Analyze elements from a website."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=['--disable-dev-shm-usage']  # Helps with memory issues
        )
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = await context.new_page()
        
        try:
            # Navigate to the URL with increased timeout
            logging.info(f"Navigating to {url}")
            await page.goto(url, timeout=60000, wait_until='domcontentloaded')
            
            # Wait for the main content to be visible
            try:
                await page.wait_for_selector('body', timeout=10000)
            except Exception as e:
                logging.warning(f"Timeout waiting for body: {e}")
            
            # Get all visible elements
            elements = await page.query_selector_all('body *')
            logging.info(f"Found {len(elements)} elements to analyze")
            
            # Analyze important elements
            important_elements = []
            for element in elements:
                try:
                    info = await extract_element_info(element)
                    
                    # Skip empty or invisible elements
                    if not info['is_visible']:
                        continue
                    
                    # Skip elements without meaningful content
                    if not info['has_text'] and not info['is_interactive'] and not info['attributes']:
                        continue
                    
                    predictions = predict_element(model, preparator, info, device)
                    
                    important_elements.append({
                        'info': info,
                        'predictions': predictions
                    })
                    
                    # Log predictions for interesting elements
                    if predictions['element_confidence'] > 0.8 or predictions['role_confidence'] > 0.8:
                        logging.info("\nAnalyzed Element:")
                        logging.info(f"Tag: {info['tag']}")
                        logging.info(f"Text: {info['inner_text'][:50]}...")
                        logging.info(f"Selector: {info['selector']}")
                        logging.info(f"Element Type: {predictions['element_type']} ({predictions['element_confidence']:.2%})")
                        logging.info(f"Semantic Role: {predictions['semantic_role']} ({predictions['role_confidence']:.2%})")
                
                except Exception as e:
                    logging.warning(f"Error analyzing element: {e}")
                    continue
            
        except Exception as e:
            logging.error(f"Error accessing website: {e}")
            raise
        
        finally:
            await browser.close()
        
        return important_elements

async def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load checkpoint
    checkpoint_path = 'models/checkpoint_epoch_10.pt'
    logging.info(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    # Analyze website
    url = 'https://www.automationexercise.com/'
    elements = await analyze_website(url, model, preparator, device)
    
    # Save results
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'website_analysis.json', 'w') as f:
        json.dump(elements, f, indent=2)
    
    logging.info(f"\nAnalyzed {len(elements)} important elements")
    logging.info(f"Results saved to {output_dir / 'website_analysis.json'}")

if __name__ == '__main__':
    asyncio.run(main())
