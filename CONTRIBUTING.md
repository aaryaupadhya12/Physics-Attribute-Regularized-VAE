# Contributing to PAR-VAE

Thank you for your interest in contributing to PAR-VAE! This document provides guidelines for contributing.

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code.

## How to Contribute

### 1. Reporting Bugs

If you find a bug, please create an issue with:
- **Clear description** of the bug
- **Steps to reproduce** the issue
- **Expected behavior** vs. actual behavior
- **Environment details** (Python version, CUDA version, hardware)
- **Reproducible example code** if possible

### 2. Suggesting Enhancements

Enhancement suggestions should include:
- **Clear use case** for the feature
- **Expected behavior**
- **Examples** of how it would work
- **Rationale** for why this would be useful

### 3. Pull Requests

#### Before You Start
1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a new branch** for your feature: `git checkout -b feature/your-feature-name`
4. **Set up development environment**: See [INSTALLATION.md](docs/INSTALLATION.md)

#### Development Requirements

Please ensure your code:
- ✓ Follows PEP 8 style guide (use `black` and `isort`)
- ✓ Includes docstrings for all functions/classes
- ✓ Has unit tests for new functionality
- ✓ Passes all existing tests
- ✓ Does not reduce code coverage
- ✓ Includes type hints where applicable

#### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Run linting
pylint src/
mypy src/

# Run tests
pytest tests/ -v --cov=src --cov-report=html
```

#### Commit Guidelines

Follow conventional commits:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Adding/updating tests
- `perf:` Performance improvement
- `chore:` Maintenance

**Example:**
```
feat(models): add physics alignment metric computation

- Implement R² calculation per physics feature
- Add visualization utilities
- Update docstrings

Closes #123
```

#### Submitting a Pull Request

1. **Push to your fork**: `git push origin feature/your-feature-name`
2. **Open a pull request** on GitHub with:
   - **Clear title** describing the change
   - **Detailed description** of changes
   - **Reference to related issues** (Closes #XXX)
   - **Screenshots/results** if applicable
3. **Respond to review comments** promptly
4. **Keep PR focused** on a single feature/fix

### 4. Documentation Contributions

Documentation improvements are highly valued:
- Update docstrings for clarity
- Add tutorial notebooks
- Improve existing documentation
- Add examples for new features

## Project Structure

```
PAR-VAE/
├── src/                    # Core source code
│   ├── models/            # VAE and regularizer implementations
│   ├── data/              # Data loading and processing
│   ├── utils/             # Utility functions
│   └── evaluation/        # Evaluation metrics and protocols
├── scripts/               # Standalone executable scripts
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit and integration tests
├── configs/               # Configuration files
├── docs/                  # Documentation
└── experiments/           # Results and experiment tracking
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow"
```

### Writing Tests

```python
# tests/test_models.py
import torch
import pytest
from src.models import VAE

def test_vae_forward_pass():
    """Test VAE forward pass produces correct shapes."""
    model = VAE(latent_dim=85)
    x = torch.randn(32, 1, 512, 512)
    
    recon, mu, logvar = model(x)
    
    assert recon.shape == x.shape
    assert mu.shape == (32, 85)
    assert logvar.shape == (32, 85)

@pytest.mark.slow
def test_vae_training():
    """Test VAE training runs without errors."""
    # ... test training ...
    pass
```

## Performance Considerations

When contributing code that involves:
- **Data loading:** Profile with large batches
- **Model training:** Test on GPU to ensure efficiency
- **Evaluation:** Ensure no unnecessary computation

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., 1.0.0)
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

## Code Review Process

1. **Automated checks:** GitHub Actions runs tests and linting
2. **Code review:** Maintainers review and request changes
3. **Approval:** At least one approving review required
4. **Merge:** Squash and merge to main branch

## Resources

- **Documentation:** See [docs/](docs/) directory
- **Issues:** Check [GitHub Issues](https://github.com/yourusername/PAR-VAE/issues)
- **Discussions:** Use GitHub Discussions for questions
- **Citation:** If building on this work, cite appropriately

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Recognized in the project README

## Questions?

Feel free to:
- Open an issue for bugs
- Use GitHub Discussions for questions
- Email maintainers directly

Thank you for contributing to PAR-VAE! 🚀
