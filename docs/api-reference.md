# API Reference

## Model API

### ElementClassifier

```python
class ElementClassifier:
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the element classifier model.
        
        Args:
            config: Model configuration dictionary
        """
        
    def predict(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Predict element type and semantic role.
        
        Args:
            element: Dictionary containing element features
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        
    def batch_predict(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict for multiple elements.
        
        Args:
            elements: List of element feature dictionaries
            
        Returns:
            List of prediction dictionaries
        """
```

### DataProcessor

```python
class DataProcessor:
    def process_html(self, html: str) -> List[Dict[str, Any]]:
        """Process HTML content and extract elements.
        
        Args:
            html: Raw HTML string
            
        Returns:
            List of processed element dictionaries
        """
        
    def normalize_element(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize element features.
        
        Args:
            element: Raw element dictionary
            
        Returns:
            Normalized element dictionary
        """
```

## Analysis API

### WebsiteAnalyzer

```python
class WebsiteAnalyzer:
    def analyze_page(self, url: str) -> Dict[str, Any]:
        """Analyze a webpage and classify elements.
        
        Args:
            url: Website URL to analyze
            
        Returns:
            Analysis results dictionary
        """
        
    def get_element_statistics(self) -> Dict[str, Any]:
        """Get element distribution statistics.
        
        Returns:
            Dictionary of element statistics
        """
```

### TestGenerator

```python
class TestGenerator:
    def generate_tests(self, analysis: Dict[str, Any]) -> str:
        """Generate Playwright tests from analysis.
        
        Args:
            analysis: Website analysis results
            
        Returns:
            Generated test code as string
        """
        
    def validate_tests(self, test_code: str) -> bool:
        """Validate generated test code.
        
        Args:
            test_code: Generated test code
            
        Returns:
            True if valid, False otherwise
        """
```

## Utility Functions

### HTML Processing

```python
def extract_features(element: BeautifulSoup) -> Dict[str, Any]:
    """Extract features from BeautifulSoup element.
    
    Args:
        element: BeautifulSoup element
        
    Returns:
        Dictionary of extracted features
    """
    
def clean_text(text: str) -> str:
    """Clean and normalize text content.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
```

### Model Utils

```python
def load_model(path: str) -> ElementClassifier:
    """Load trained model from path.
    
    Args:
        path: Path to model checkpoint
        
    Returns:
        Loaded ElementClassifier instance
    """
    
def evaluate_model(model: ElementClassifier, 
                  test_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate model performance.
    
    Args:
        model: ElementClassifier instance
        test_data: List of test examples
        
    Returns:
        Dictionary of evaluation metrics
    """
```

### Test Generation Utils

```python
def create_test_suite(tests: List[str]) -> str:
    """Create a test suite from individual tests.
    
    Args:
        tests: List of test code strings
        
    Returns:
        Combined test suite code
    """
    
def format_test_code(code: str) -> str:
    """Format generated test code.
    
    Args:
        code: Raw test code string
        
    Returns:
        Formatted test code string
    """
```

## Configuration Objects

### ModelConfig

```python
@dataclass
class ModelConfig:
    bert_model: str
    max_length: int
    hidden_size: int
    num_labels: Dict[str, int]
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    warmup_steps: int
    weight_decay: float
```

### TestConfig

```python
@dataclass
class TestConfig:
    timeout: int
    retries: int
    headless: bool
    viewport: Dict[str, int]
```

## Error Types

```python
class ModelError(Exception):
    """Base class for model-related errors."""
    pass

class DataProcessingError(Exception):
    """Error in data processing pipeline."""
    pass

class TestGenerationError(Exception):
    """Error in test generation process."""
    pass
```

## Constants

```python
# Element Types
ELEMENT_TYPES = {
    "INTERACTIVE": 0,
    "NAVIGATION": 1,
    "CONTENT": 2,
    "MEDIA": 3,
    "CONTAINER": 4
}

# Semantic Roles
SEMANTIC_ROLES = {
    "BUTTON": 0,
    "LINK": 1,
    "INPUT": 2,
    "TEXT": 3
}

# Model Parameters
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
```
