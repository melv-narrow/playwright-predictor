# Configuration Guide

## Environment Setup

### 1. Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. GPU Configuration
```python
# config/gpu_config.py
gpu_config = {
    "use_gpu": True,
    "mixed_precision": True,
    "memory_efficient": True
}
```

## Model Configuration

### 1. Training Configuration
```python
# config/training_config.py
training_config = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 10,
    "warmup_steps": 1000,
    "weight_decay": 0.01,
    "gradient_accumulation": 1
}
```

### 2. Model Architecture
```python
# config/model_config.py
model_config = {
    "bert_model": "bert-base-uncased",
    "max_length": 512,
    "hidden_size": 768,
    "num_labels": {
        "element_type": 5,
        "semantic_role": 4
    }
}
```

## Data Collection Configuration

### 1. Web Scraping
```yaml
# config/scraping_config.yaml
scraping:
  max_depth: 3
  timeout: 30
  headers:
    User-Agent: "Mozilla/5.0..."
  ignore_patterns:
    - "*.pdf"
    - "*.jpg"
```

### 2. Data Processing
```yaml
# config/processing_config.yaml
processing:
  text_cleaning: true
  normalize_attributes: true
  remove_scripts: true
  min_text_length: 1
```

## Test Generation Configuration

### 1. Test Strategy
```yaml
# config/test_strategy.yaml
strategy:
  coverage:
    element_types: true
    interactive: true
    visible: true
  priorities:
    - forms
    - navigation
    - content
```

### 2. Playwright Configuration
```javascript
// playwright.config.ts
export default {
  timeout: 30000,
  retries: 2,
  workers: 1,
  reporter: 'html',
  use: {
    headless: true,
    viewport: { width: 1280, height: 720 },
    trace: 'on-first-retry',
  },
};
```

## Logging Configuration

### 1. Python Logging
```python
# config/logging_config.py
logging_config = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "app.log",
            "formatter": "standard"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["file"]
    }
}
```

### 2. Debug Configuration
```yaml
# config/debug_config.yaml
debug:
  verbose: true
  save_artifacts: true
  trace_calls: false
```

## Performance Tuning

### 1. Memory Management
```python
# config/memory_config.py
memory_config = {
    "max_elements": 10000,
    "batch_size": 32,
    "prefetch_factor": 2,
    "num_workers": 4
}
```

### 2. Optimization
```python
# config/optimization_config.py
optimization_config = {
    "mixed_precision": True,
    "gradient_checkpointing": False,
    "optimize_memory_use": True
}
```

## Environment Variables

Create a `.env` file in the project root:
```bash
# .env
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=true
CUDA_VISIBLE_DEVICES=0
```

## Configuration Best Practices

1. **Version Control**
   - Keep sensitive data out of version control
   - Use environment variables for secrets
   - Document all configuration options

2. **Validation**
   - Validate configuration at startup
   - Provide sensible defaults
   - Log configuration changes

3. **Maintenance**
   - Regular updates
   - Backward compatibility
   - Clear documentation
