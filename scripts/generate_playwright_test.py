"""
Generate Playwright tests using the enhanced model.
"""

import torch
from pathlib import Path
import logging
from transformers import AutoTokenizer
from model import ElementClassifier
from finetune_playwright import EnhancedElementClassifier, TestGenerationHead

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlaywrightTestGenerator:
    def __init__(self, model_path: str = 'models/enhanced_model.pt', device: str = 'cuda'):
        self.device = device
        
        # Load checkpoint
        logger.info("Loading model...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create base model
        config = checkpoint['config']
        base_model = ElementClassifier(
            num_element_types=config['num_element_types'],
            num_semantic_roles=config['num_semantic_roles'],
            device=device
        )
        
        # Create enhanced model
        self.model = EnhancedElementClassifier(base_model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        # Pattern mapping
        self.pattern_map = {
            0: 'assertion',
            1: 'navigation',
            2: 'fixture',
            3: 'wait_strategy',
            4: 'form_interaction',
            5: 'page_object',
            6: 'general'
        }
        self.reverse_pattern_map = {v: k for k, v in self.pattern_map.items()}
    
    def generate_test(self, element_info: dict) -> dict:
        """Generate a Playwright test for a given element."""
        # Prepare input
        element_desc = f"""
Element Type: {element_info.get('element_type', '')}
Semantic Role: {element_info.get('semantic_role', '')}
Text Content: {element_info.get('text', '')}
Selector: {element_info.get('selector', '')}
Attributes: {element_info.get('attributes', {})}
Interactive: {element_info.get('is_interactive', False)}
Visible: {element_info.get('is_visible', True)}
""".strip()
        
        # Tokenize
        inputs = self.tokenizer(
            element_desc,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model(
                inputs['input_ids'],
                inputs['attention_mask'],
                torch.zeros(1, 3).to(self.device)  # Dummy numerical features
            )
            
            # Get pattern prediction
            pattern_logits = outputs['pattern']
            pattern_probs = torch.softmax(pattern_logits, dim=1)[0]
            
            # Convert to numpy for easier manipulation
            probs = pattern_probs.cpu().numpy()
            
            # Apply heuristics to adjust pattern selection
            element_type = element_info.get('element_type', '').lower()
            is_interactive = element_info.get('is_interactive', False)
            attributes = element_info.get('attributes', {})
            
            # Boost form_interaction for interactive elements
            if is_interactive:
                probs[self.reverse_pattern_map['form_interaction']] *= 4.0
                
                # For buttons and inputs, form_interaction should be even higher
                if element_type in ['button', 'input', 'textarea', 'select']:
                    probs[self.reverse_pattern_map['form_interaction']] *= 3.0
                    
                    # Extra boost for input fields
                    if element_type == 'input':
                        probs[self.reverse_pattern_map['form_interaction']] *= 1.5
                    
                # For submit buttons specifically
                if element_type == 'button' and attributes.get('type') == 'submit':
                    probs[self.reverse_pattern_map['form_interaction']] *= 2.0
            
            # Boost navigation for links and buttons that aren't form submissions
            if element_type in ['a', 'link'] or (element_type == 'button' and attributes.get('type') != 'submit'):
                probs[self.reverse_pattern_map['navigation']] *= 3.0
                
                # Extra boost for elements with href
                if 'href' in attributes:
                    probs[self.reverse_pattern_map['navigation']] *= 2.0
            
            # Boost wait_strategy for dynamic elements
            if any(attr in attributes for attr in ['data-loading', 'data-dynamic', 'async']):
                probs[self.reverse_pattern_map['wait_strategy']] *= 2.0
            
            # Boost page_object for container elements
            if element_type in ['div', 'section', 'form', 'nav']:
                probs[self.reverse_pattern_map['page_object']] *= 2.0
            
            # Reduce general pattern and assertion probabilities
            probs[self.reverse_pattern_map['general']] *= 0.5
            if element_type in ['input', 'textarea', 'select']:
                probs[self.reverse_pattern_map['assertion']] *= 0.5
            
            # Normalize probabilities
            probs = probs / probs.sum()
            
            # Get final pattern
            pattern_idx = probs.argmax()
            pattern = self.pattern_map[pattern_idx]
            
            # Generate test code
            test_code = self._generate_test_code(element_info, pattern)
            
            return {
                'pattern': pattern,
                'test_code': test_code,
                'confidence': float(probs[pattern_idx]),
                'all_patterns': {
                    self.pattern_map[i]: float(prob)
                    for i, prob in enumerate(probs)
                }
            }
    
    def _generate_test_code(self, element_info: dict, pattern: str) -> str:
        """Generate test code based on element info and pattern."""
        selector = element_info.get('selector', '')
        text = element_info.get('text', '')
        element_type = element_info.get('element_type', '').lower()
        attributes = element_info.get('attributes', {})
        
        if pattern == 'assertion':
            return (
                'test(\'should verify element content\', async ({ page }) => {\n'
                f'    const element = page.locator(\'{selector}\');\n'
                '    await expect(element).toBeVisible();\n'
                f'    await expect(element).toHaveText(\'{text}\');\n'
                '});'
            )
            
        elif pattern == 'navigation':
            # Handle navigation with href if available
            href = attributes.get('href', '')
            url_comment = '' if href else '  // Add URL'
            
            # Format the URL properly
            if href:
                if href.startswith('http'):
                    goto_url = f"'{href}'"
                else:
                    # Handle relative URLs
                    base_path = href if href.startswith('/') else f'/{href}'
                    goto_url = f'`${{baseUrl}}{base_path}`'
            else:
                goto_url = "''"
            
            # Generate navigation test with URL verification
            return (
                'test(\'should navigate to element\', async ({ page, baseUrl }) => {\n'
                f'    await page.goto({goto_url}){url_comment};\n'
                f'    const element = page.locator(\'{selector}\');\n'
                '    await expect(element).toBeVisible();\n'
                '    await element.click();\n'
                '    // Verify navigation\n'
                f'    await page.waitForURL({goto_url});\n'
                '    await expect(page).toHaveURL(new RegExp(\'.*\'));\n'
                '});'
            )
            
        elif pattern == 'form_interaction':
            # Handle different form elements
            if element_type == 'button':
                return (
                    'test(\'should click button\', async ({ page }) => {\n'
                    f'    const button = page.locator(\'{selector}\');\n'
                    '    await expect(button).toBeVisible();\n'
                    '    await expect(button).toBeEnabled();\n'
                    '    await button.click();\n'
                    '});'
                )
            elif element_type == 'input':
                input_type = attributes.get('type', 'text')
                if input_type == 'checkbox':
                    return (
                        'test(\'should toggle checkbox\', async ({ page }) => {\n'
                        f'    const checkbox = page.locator(\'{selector}\');\n'
                        '    await expect(checkbox).toBeVisible();\n'
                        '    await checkbox.check();\n'
                        '    await expect(checkbox).toBeChecked();\n'
                        '});'
                    )
                elif input_type == 'radio':
                    return (
                        'test(\'should select radio button\', async ({ page }) => {\n'
                        f'    const radio = page.locator(\'{selector}\');\n'
                        '    await expect(radio).toBeVisible();\n'
                        '    await radio.check();\n'
                        '    await expect(radio).toBeChecked();\n'
                        '});'
                    )
                else:
                    placeholder = attributes.get('placeholder', 'Test input')
                    return (
                        'test(\'should fill input field\', async ({ page }) => {\n'
                        f'    const input = page.locator(\'{selector}\');\n'
                        '    await expect(input).toBeVisible();\n'
                        f'    await input.fill(\'{placeholder}\');\n'
                        f'    await expect(input).toHaveValue(\'{placeholder}\');\n'
                        '});'
                    )
            elif element_type == 'select':
                return (
                    'test(\'should select option\', async ({ page }) => {\n'
                    f'    const select = page.locator(\'{selector}\');\n'
                    '    await expect(select).toBeVisible();\n'
                    '    await select.selectOption({ index: 0 });\n'
                    '    await expect(select).toHaveValue();\n'
                    '});'
                )
            elif element_type == 'textarea':
                return (
                    'test(\'should fill textarea\', async ({ page }) => {\n'
                    f'    const textarea = page.locator(\'{selector}\');\n'
                    '    await expect(textarea).toBeVisible();\n'
                    '    await textarea.fill(\'Test content\');\n'
                    '    await expect(textarea).toHaveValue(\'Test content\');\n'
                    '});'
                )
            else:
                return (
                    'test(\'should interact with form element\', async ({ page }) => {\n'
                    f'    const element = page.locator(\'{selector}\');\n'
                    '    await expect(element).toBeVisible();\n'
                    '    await element.click();\n'
                    '});'
                )
            
        elif pattern == 'wait_strategy':
            return (
                'test(\'should wait for element\', async ({ page }) => {\n'
                f'    const element = page.locator(\'{selector}\');\n'
                '    await element.waitFor({ state: \'visible\' });\n'
                '    await expect(element).toBeVisible();\n'
                '});'
            )
            
        elif pattern == 'page_object':
            class_name = ''.join(word.capitalize() for word in text.split())
            return (
                f'class {class_name}Page {{\n'
                '    constructor(page) {\n'
                '        this.page = page;\n'
                f'        this.element = page.locator(\'{selector}\');\n'
                '    }\n'
                '    \n'
                '    async isVisible() {\n'
                '        await expect(this.element).toBeVisible();\n'
                '    }\n'
                '    \n'
                '    async getText() {\n'
                '        return await this.element.textContent();\n'
                '    }\n'
                '}\n'
                '\n'
                'test(\'should use page object\', async ({ page }) => {\n'
                f'    const pageObj = new {class_name}Page(page);\n'
                '    await pageObj.isVisible();\n'
                f'    expect(await pageObj.getText()).toBe(\'{text}\');\n'
                '});'
            )
            
        else:  # general or fixture
            return (
                'test(\'should handle element\', async ({ page }) => {\n'
                f'    const element = page.locator(\'{selector}\');\n'
                '    await expect(element).toBeVisible();\n'
                '    // Add more test steps here\n'
                '});'
            )

def main():
    # Example usage
    generator = PlaywrightTestGenerator()
    
    # Test different element types
    elements = [
        {
            'element_type': 'button',
            'semantic_role': 'button',
            'text': 'Submit Form',
            'selector': '#submit-btn',
            'attributes': {'type': 'submit', 'class': 'btn btn-primary'},
            'is_interactive': True,
            'is_visible': True
        },
        {
            'element_type': 'input',
            'semantic_role': 'textbox',
            'text': '',
            'selector': '#username',
            'attributes': {'type': 'text', 'placeholder': 'Enter username', 'class': 'form-control'},
            'is_interactive': True,
            'is_visible': True
        },
        {
            'element_type': 'a',
            'semantic_role': 'link',
            'text': 'Sign Up',
            'selector': '.signup-link',
            'attributes': {'href': '/signup', 'class': 'nav-link'},
            'is_interactive': True,
            'is_visible': True
        }
    ]
    
    # Generate tests for each element
    for element in elements:
        print(f"\nTesting {element['element_type']}: {element['text'] or element['selector']}")
        result = generator.generate_test(element)
        
        print(f"\nDetected Pattern: {result['pattern']} (Confidence: {result['confidence']:.2%})")
        print("\nAll Pattern Probabilities:")
        for pattern, prob in sorted(result['all_patterns'].items(), key=lambda x: x[1], reverse=True):
            print(f"{pattern}: {prob:.2%}")
        print("\nGenerated Test:")
        print(result['test_code'])
        print("\n" + "="*80)

if __name__ == '__main__':
    main()
