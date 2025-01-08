"""
Generate Playwright tests using the trained model.
"""

import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer
from typing import Dict, List
import json
from finetune_playwright import EnhancedElementClassifier
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlaywrightTestGenerator:
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        self.model = EnhancedElementClassifier(
            base_model='microsoft/codebert-base',
            num_patterns=7,
            num_test_patterns=7
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def generate_test(self, instruction: str, html_content: str) -> Dict:
        """Generate a Playwright test for the given input."""
        # Prepare input
        inputs = self.tokenizer(
            instruction + ' ' + html_content,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Get predictions and pattern
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            pattern_logits = outputs.pattern_logits
            test_logits = outputs.test_logits
        
        # Generate test based on pattern
        test_code = self._generate_test_code(
            html_content,
            pattern_logits,
            test_logits
        )
        
        return {
            'test_code': test_code,
            'pattern_logits': pattern_logits.cpu().numpy().tolist(),
            'test_logits': test_logits.cpu().numpy().tolist()
        }
    
    def _generate_test_code(
        self,
        html_content: str,
        pattern_logits: torch.Tensor,
        test_logits: torch.Tensor
    ) -> str:
        """Generate Playwright test code based on the model outputs."""
        # Extract relevant elements from HTML
        input_fields = re.findall(r'<input[^>]*>', html_content)
        buttons = re.findall(r'<button[^>]*>[^<]*</button>', html_content)
        
        # Get element IDs and types
        elements = []
        for field in input_fields:
            id_match = re.search(r'id="([^"]*)"', field)
            type_match = re.search(r'type="([^"]*)"', field)
            if id_match:
                elements.append({
                    'id': id_match.group(1),
                    'type': type_match.group(1) if type_match else 'text'
                })
        
        # Generate test code based on pattern
        pattern_idx = torch.argmax(pattern_logits, dim=-1).item()
        
        # Basic test structure
        test_code = [
            "import { test, expect } from '@playwright/test';",
            "",
            "test('Generated Test', async ({ page }) => {",
        ]
        
        # Add test steps based on elements and pattern
        if pattern_idx == 4:  # form_interaction
            for elem in elements:
                if elem['type'] == 'text':
                    test_code.append(f"  await page.fill('#{elem['id']}', 'test_{elem['id']}');")
                elif elem['type'] == 'password':
                    test_code.append(f"  await page.fill('#{elem['id']}', 'TestPassword123!');")
            
            if buttons:
                test_code.append(f"  await page.click('button:has-text(\"Submit\")');")
                test_code.append(f"  await expect(page).toHaveURL(/.*success/);")
        
        elif pattern_idx == 1:  # navigation
            test_code.extend([
                "  await page.goto('http://example.com');",
                "  await expect(page).toHaveTitle(/.*Home.*/);",
            ])
        
        elif pattern_idx == 0:  # assertion
            for elem in elements:
                test_code.append(f"  await expect(page.locator('#{elem['id']}')).toBeVisible();")
        
        # Close test
        test_code.append("});")
        
        return '\n'.join(test_code)

def main():
    # Initialize generator
    generator = PlaywrightTestGenerator(
        model_path='models/playwright_enhanced_model.pt'
    )
    
    # Example input
    example = {
        'instruction': 'Generate a test for login functionality',
        'html_content': '''
        <form>
            <input type="text" id="username" placeholder="Username">
            <input type="password" id="password" placeholder="Password">
            <button type="submit">Login</button>
        </form>
        '''
    }
    
    # Generate test
    result = generator.generate_test(
        example['instruction'],
        example['html_content']
    )
    
    # Print generated test
    logger.info("\nGenerated Test:")
    logger.info(result['test_code'])
    
    # Save test to file
    output_dir = Path('generated_tests')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'login_test.spec.ts', 'w') as f:
        f.write(result['test_code'])
    
    logger.info(f"\nTest saved to {output_dir / 'login_test.spec.ts'}")

if __name__ == '__main__':
    main() 