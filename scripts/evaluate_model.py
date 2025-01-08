"""
Evaluate the trained model's ability to generate Playwright tests.
"""

import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer
from typing import Dict, List
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from finetune_playwright import EnhancedElementClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlaywrightEvaluator:
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
    
    def predict_pattern(self, instruction: str, input_text: str) -> Dict:
        """Predict the test pattern for given input."""
        # Prepare input
        inputs = self.tokenizer(
            instruction + ' ' + input_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            pattern_probs = torch.softmax(outputs.pattern_logits, dim=-1)
            pattern_idx = torch.argmax(pattern_probs, dim=-1).item()
        
        return {
            'pattern': self.pattern_map[pattern_idx],
            'confidence': pattern_probs[0][pattern_idx].item(),
            'all_probs': {
                self.pattern_map[i]: prob.item()
                for i, prob in enumerate(pattern_probs[0])
            }
        }
    
    def evaluate_test_set(self, test_data_path: str):
        """Evaluate model performance on test set."""
        logger.info(f"Evaluating test set from {test_data_path}")
        
        # Load test data
        with open(test_data_path) as f:
            test_data = [json.loads(line) for line in f]
        
        # Collect predictions and true labels
        y_true = []
        y_pred = []
        confidences = []
        
        for example in tqdm(test_data, desc="Evaluating"):
            prediction = self.predict_pattern(
                example['instruction'],
                example['input']
            )
            true_pattern = example['metadata']['pattern']
            
            y_true.append(true_pattern)
            y_pred.append(prediction['pattern'])
            confidences.append(prediction['confidence'])
        
        # Generate classification report
        report = classification_report(y_true, y_pred)
        logger.info("\nClassification Report:\n" + report)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(self.pattern_map.values()))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            xticklabels=list(self.pattern_map.values()),
            yticklabels=list(self.pattern_map.values())
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Pattern')
        plt.xlabel('Predicted Pattern')
        plt.tight_layout()
        plt.savefig('evaluation_results/confusion_matrix.png')
        
        # Analyze confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20)
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('evaluation_results/confidence_distribution.png')
        
        # Save detailed results
        results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'average_confidence': np.mean(confidences),
            'predictions': [
                {
                    'input': example['input'],
                    'true_pattern': example['metadata']['pattern'],
                    'predicted_pattern': pred,
                    'confidence': conf
                }
                for example, pred, conf in zip(test_data, y_pred, confidences)
            ]
        }
        
        with open('evaluation_results/detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    # Create evaluation directory
    Path('evaluation_results').mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = PlaywrightEvaluator(
        model_path='models/playwright_enhanced_model.pt'
    )
    
    # Example of single prediction
    example = {
        'instruction': 'Generate a test for login functionality',
        'input': '''
        <form>
            <input type="text" id="username" placeholder="Username">
            <input type="password" id="password" placeholder="Password">
            <button type="submit">Login</button>
        </form>
        '''
    }
    
    prediction = evaluator.predict_pattern(
        example['instruction'],
        example['input']
    )
    logger.info(f"\nSingle Prediction Example:")
    logger.info(f"Input: {example['input']}")
    logger.info(f"Predicted Pattern: {prediction['pattern']}")
    logger.info(f"Confidence: {prediction['confidence']:.4f}")
    logger.info("Pattern Probabilities:")
    for pattern, prob in prediction['all_probs'].items():
        logger.info(f"  {pattern}: {prob:.4f}")
    
    # Evaluate on test set
    test_data_path = 'training_data/playwright_docs/training/test.jsonl'
    results = evaluator.evaluate_test_set(test_data_path)
    
    logger.info("\nEvaluation complete! Results saved in evaluation_results/")

if __name__ == '__main__':
    main() 