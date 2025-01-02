# AI-Powered Playwright Test Generator

An intelligent test automation framework that analyzes websites and automatically generates Playwright test suites using machine learning.

## Features

- Website analysis and data collection
- ML-powered element identification
- Automated Playwright test generation
- Configurable test strategies
- Comprehensive logging and error handling

## Project Structure

```
playwright-ai/
├── src/
│   ├── ai/                 # AI/ML components
│   │   ├── models/        # ML model definitions
│   │   ├── training/      # Training scripts
│   │   └── inference/     # Inference engine
│   ├── scraper/           # Web scraping components
│   ├── test_generator/    # Test generation logic
│   └── utils/             # Shared utilities
├── tests/                 # Framework tests
├── config/                # Configuration files
├── examples/              # Usage examples
└── generated_tests/       # Output directory for generated tests
```

## Requirements

- Python 3.9+
- Node.js 16+
- TypeScript 4.8+
- Playwright
- PyTorch
- transformers
- Beautiful Soup 4

## Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Node.js dependencies:
   ```bash
   npm install
   ```

## Usage

1. Configure your target website in `config/sites.yaml`
2. Run the test generator:
   ```bash
   python src/main.py generate --config config/sites.yaml
   ```
3. Execute generated tests:
   ```bash
   npx playwright test
   ```

## Documentation

Detailed documentation is available in the `docs/` directory:
- [Architecture Overview](docs/architecture.md)
- [ML Model Details](docs/ml-model.md)
- [Configuration Guide](docs/configuration.md)
- [API Reference](docs/api-reference.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.