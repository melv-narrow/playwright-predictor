"""
Generate Playwright tests using the enhanced model.
"""

import torch
from pathlib import Path
import logging
from transformers import AutoModel, AutoTokenizer
from model import ElementClassifier
from finetune_playwright import EnhancedElementClassifier, TestGenerationHead
from typing import List, Dict
import torch.nn as nn
import asyncio
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATTERN_TYPES = ['click', 'navigation', 'input', 'checkbox']
TEST_PATTERNS = [
    "await expect({locator}).toHaveAttribute('aria-expanded', 'true');",
    "await expect({locator}).toHaveAttribute('aria-checked', 'true');",
    "await expect({locator}).toHaveValue('');",
    "await expect({locator}).not.toHaveValue('');"
]

class EnhancedElementClassifier(nn.Module):
    """Enhanced model that combines element classification and test generation."""
    def __init__(self, base_model: str, num_patterns: int, num_test_patterns: int):
        super().__init__()
        # Don't initialize transformer here - we'll load it from state dict
        self.transformer = None
        
        # Processors for numerical features (updated dimensions)
        self.numerical_processor = nn.Sequential(
            nn.Linear(3, 192),  # Changed from 128 to 192
            nn.ReLU()
        )
        
        # Combined processing (updated dimensions)
        self.combined_processor = nn.Sequential(
            nn.Linear(960, 768),  # Adjusted to match input dimensions
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 384)  # Changed from 256 to 384
        )
        
        # Classification heads (updated dimensions)
        self.element_type_head = nn.Linear(384, 5)  # Changed input and output sizes
        self.semantic_role_head = nn.Linear(384, 6)  # Changed input and output sizes
        
        # Test generation head
        self.test_head = TestGenerationHead()

    def forward(self, input_ids=None, attention_mask=None, numerical_features=None):
        """Forward pass through the model."""
        if self.transformer is None:
            raise ValueError("Transformer not initialized")
            
        # Get transformer embeddings
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        text_features = transformer_output.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Process numerical features
        if numerical_features is not None:
            num_features = self.numerical_processor(numerical_features)
            # Concatenate text and numerical features
            combined_features = torch.cat([text_features, num_features], dim=1)
        else:
            # If no numerical features, pad with zeros
            batch_size = text_features.size(0)
            num_features = torch.zeros(batch_size, 192, device=text_features.device)
            combined_features = torch.cat([text_features, num_features], dim=1)
            
        # Process combined features
        processed_features = self.combined_processor(combined_features)
        
        # Get predictions from each head
        element_type_logits = self.element_type_head(processed_features)
        semantic_role_logits = self.semantic_role_head(processed_features)
        
        # Generate test cases
        test_output = self.test_head(processed_features)
        
        return {
            'element_type': element_type_logits,
            'semantic_role': semantic_role_logits,
            'test_output': test_output
        }

class PlaywrightTestGenerator:
    def __init__(self, model_path: str = 'models/enhanced_model.pt', device: str = 'cuda'):
        self.device = device
        self.model = ElementClassifier(
            num_element_types=5,
            num_semantic_roles=6
        )
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        # Update pattern maps to match model output dimensions
        self.reverse_pattern_map = {
            'form_interaction': 0, 
            'navigation': 1, 
            'wait_strategy': 2, 
            'page_object': 3, 
            'general': 4
        }
        self.pattern_map = {
            0: 'form_interaction',
            1: 'navigation',
            2: 'wait_strategy',
            3: 'page_object',
            4: 'general'
        }
        self.load_model(model_path)

    def generate_test(self, element_info: dict) -> dict:
        """Generate a Playwright test for a given element."""
        logger.info(f"Generating tests from website: {element_info.get('url', 'unknown')}")
        
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
            print(f"Outputs type: {type(outputs)}, Outputs: {outputs}")
            logger.info(f"Outputs structure: {outputs}")
            logger.info(f"Outputs: {outputs}")
            
            # Get pattern prediction
            pattern_logits, test_logits = outputs
            logger.info(f"Test logits: {test_logits}")
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
                if element_type in ['button', 'input', 'textarea', 'select']:
                    probs[self.reverse_pattern_map['form_interaction']] *= 3.0
                    if element_type == 'input':
                        probs[self.reverse_pattern_map['form_interaction']] *= 1.5
                if element_type == 'button' and attributes.get('type') == 'submit':
                    probs[self.reverse_pattern_map['form_interaction']] *= 2.0
            
            # Boost navigation for links and buttons that aren't form submissions
            if element_type in ['a', 'link'] or (element_type == 'button' and attributes.get('type') != 'submit'):
                probs[self.reverse_pattern_map['navigation']] *= 3.0
                if 'href' in attributes:
                    probs[self.reverse_pattern_map['navigation']] *= 2.0
            
            # Boost wait_strategy for dynamic elements
            if any(attr in attributes for attr in ['data-loading', 'data-dynamic', 'async']):
                probs[self.reverse_pattern_map['wait_strategy']] *= 2.0
            
            # Boost page_object for container elements
            if element_type in ['div', 'section', 'form', 'nav']:
                probs[self.reverse_pattern_map['page_object']] *= 2.0
            
            # Reduce general pattern probability
            probs[self.reverse_pattern_map['general']] *= 0.5
            
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
                f'    logger.info("Navigating to URL: {goto_url}")\n'
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

    def load_model(self, model_path):
        """Load the model from the specified path."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def load_model():
    """Load the fine-tuned model for test generation."""
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        # Initialize the model architecture
        model = EnhancedElementClassifier(
            base_model='microsoft/codebert-base',
            num_patterns=5,
            num_test_patterns=6
        )
        
        # Load the trained weights
        checkpoint = torch.load('models/enhanced_model.pt', map_location='cpu')
        
        # Remove 'base_model.' prefix from state dict keys
        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('base_model.'):
                new_key = key.replace('base_model.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
                
        # Initialize transformer from the state dict
        model.transformer = AutoModel.from_pretrained('microsoft/codebert-base')
        
        # Load the state dict
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        logger.info("Successfully loaded fine-tuned model")
        return {'model': model, 'tokenizer': tokenizer}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None  # Fall back to rule-based generation

def generate_test_case(model_data, element_info: dict) -> str:
    """Generate a test case based on element information using the fine-tuned model."""
    if not element_info.get("is_visible", True) or not element_info.get("is_enabled", True):
        return None
        
    # Prepare element info for model input
    element_text = prepare_element_text(element_info)
    
    # Tokenize the input
    inputs = model_data['tokenizer'](
        element_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Get model predictions
    with torch.no_grad():
        outputs = model_data['model'](
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        print(f"Outputs type: {type(outputs)}, Outputs: {outputs}")
        
    # Get pattern predictions
    pattern_logits, test_logits = outputs
    logger.info(f"Test logits: {test_logits}")
    pattern_probs = torch.sigmoid(pattern_logits)
    test_probs = torch.sigmoid(test_logits)
    
    # Select patterns with high confidence
    pattern_indices = torch.where(pattern_probs > 0.5)[1]
    test_indices = torch.where(test_probs > 0.5)[1]
    
    # Generate test based on predicted patterns
    test_name = generate_test_name(element_info)
    test_content = []
    
    # Generate locator
    locator = generate_locator(element_info)
    if not locator:
        return None
        
    # Add predicted test patterns
    test_content.extend(generate_model_based_actions(
        element_info,
        locator,
        pattern_indices.tolist(),
        test_indices.tolist()
    ))
    
    if not test_content:
        return None
        
    # Format the test case
    test_case = f"""
    test('{test_name}', async ({{ page }}) => {{
        {chr(10).join(f"        {line}" for line in test_content)}
    }});"""
    
    return test_case

def prepare_element_text(element_info: dict) -> str:
    """Prepare element information as text input for the model."""
    properties = []
    
    # Add tag name
    properties.append(f"tag:{element_info.get('tag_name', '')}")
    
    # Add element type
    if element_info.get('type'):
        properties.append(f"type:{element_info['type']}")
        
    # Add role
    if element_info.get('role'):
        properties.append(f"role:{element_info['role']}")
        
    # Add other important attributes
    for attr in ['id', 'class', 'name', 'placeholder', 'href', 'aria-label', 'data-testid']:
        if element_info.get(attr):
            properties.append(f"{attr}:{element_info[attr]}")
            
    # Add text content
    if element_info.get('text'):
        properties.append(f"text:{element_info['text']}")
        
    return " ".join(properties)

def generate_model_based_actions(element_info: dict, locator: str, pattern_indices: List[int], test_indices: List[int]) -> List[str]:
    """Generate test actions based on model predictions."""
    actions = []
    
    # Always start with visibility check
    actions.append(f"await expect({locator}).toBeVisible();")
    
    # Add actions based on predicted patterns
    for pattern_idx in pattern_indices:
        pattern_type = PATTERN_TYPES[pattern_idx]
        
        if pattern_type == "click":
            actions.append(f"await {locator}.click();")
        elif pattern_type == "navigation":
            if element_info.get("href"):
                href = element_info["href"]
                if href.startswith("/"):
                    href = f"https://github.com{href}"
                actions.append(f"await Promise.all([")
                actions.append(f"    page.waitForNavigation(),")
                actions.append(f"    {locator}.click()")
                actions.append(f"]);")
                actions.append(f"await expect(page).toHaveURL('{href}');")
        elif pattern_type == "input":
            test_value = get_test_value(element_info)
            actions.append(f"await {locator}.click();")
            actions.append(f"await {locator}.fill('{test_value}');")
            actions.append(f"await expect({locator}).toHaveValue('{test_value}');")
        elif pattern_type == "checkbox":
            actions.append(f"await {locator}.check();")
            actions.append(f"await expect({locator}).toBeChecked();")
            actions.append(f"await {locator}.uncheck();")
            actions.append(f"await expect({locator}).not.toBeChecked();")
            
    # Add additional test patterns
    for test_idx in test_indices:
        test_pattern = TEST_PATTERNS[test_idx]
        if test_pattern not in [a.strip() for a in actions]:
            actions.append(test_pattern.format(locator=locator))
            
    return actions

def generate_test_name(element_info: dict) -> str:
    """Generate a descriptive test name based on element info."""
    element_type = element_info.get("tag_name", "element")
    
    # Try to get a descriptive name from various attributes
    descriptors = [
        element_info.get("aria-label"),
        element_info.get("text", "").strip(),
        element_info.get("placeholder"),
        element_info.get("name"),
        element_info.get("id")
    ]
    
    descriptor = next((d for d in descriptors if d), "unknown")
    descriptor = descriptor.replace("'", "").replace('"', "")[:30]  # Truncate long names
    
    return f"should handle {element_type} - {descriptor}"

def generate_locator(element_info: dict) -> str:
    """Generate the most reliable locator for the element."""
    # Try different locator strategies in order of reliability
    if element_info.get("data-testid"):
        return f"page.getByTestId('{element_info['data-testid']}')"
    elif element_info.get("aria-label"):
        return f"page.getByRole('{element_info.get('role', 'generic')}', {{ name: '{element_info['aria-label']}' }})"
    elif element_info.get("id"):
        return f"page.locator('#{element_info['id']}')"
    elif element_info.get("name"):
        return f"page.getByRole('{element_info.get('role', 'generic')}', {{ name: '{element_info['name']}' }})"
    elif element_info.get("text"):
        text = element_info["text"].strip().replace("'", "\\'")
        return f"page.getByText('{text}')"
    else:
        # Fallback to CSS selector
        return f"page.locator('{element_info['selector']}')"

def get_test_value(element_info: dict) -> str:
    """Generate appropriate test value based on element type."""
    element_type = element_info.get("type", "").lower()
    placeholder = element_info.get("placeholder", "").lower()
    
    if element_type == "email":
        return "test@example.com"
    elif element_type == "password":
        return "TestPassword123!"
    elif "search" in element_type or "search" in placeholder:
        return "test search query"
    else:
        return "test value"

def is_interactive_element(element_info: dict) -> bool:
    """Determine if an element is interactive and worth testing."""
    tag_name = element_info.get("tag_name", "").lower()
    element_type = element_info.get("type", "").lower()
    role = element_info.get("role", "").lower()
    
    # Always include form elements
    if tag_name in ["input", "select", "textarea"]:
        return True
        
    # Include buttons and links with href
    if tag_name == "button" or role == "button":
        return True
    if tag_name == "a" and element_info.get("href"):
        return True
        
    # Include elements with click handlers or specific roles
    if role in ["button", "link", "menuitem", "tab", "checkbox", "radio"]:
        return True
        
    return False

async def main():
    # Launch the browser and open a new page
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto('https://github.com/')

        # Analyze the page and gather element information
        elements = await page.query_selector_all('a, button, input')  # Example selectors
        element_info_list = []

        for element in elements:
            tag_name = await element.evaluate('el => el.tagName')
            selector = await element.evaluate('el => el.selector')
            is_visible = await element.is_visible()
            element_info_list.append({
                'element_type': tag_name,
                'selector': selector,
                'is_visible': is_visible,
                'text': await element.inner_text(),
                'attributes': await element.evaluate('el => el.attributes'),
            })

        logger.info(f"Element info list: {element_info_list}")
        
        output_file = "generated_tests.spec.ts"

        with open(output_file, "w") as file:
            file.write("// Generated Playwright Tests\n\n")

            # Generate tests for each element
            generator = PlaywrightTestGenerator()
            for element_info in element_info_list:
                result = generator.generate_test(element_info)
                logger.info(f"Generated test result: {result}")
                file.write(result['test_code'] + "\n\n")

        logger.info(f"Generated tests saved to {output_file}")

        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
