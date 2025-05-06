# Contributing to Decision Jungles

Thank you for your interest in contributing to the Decision Jungles project! We welcome contributions from the community to help improve and expand this scikit-learn compatible implementation of Decision Jungles.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please be kind and courteous to other community members, and help foster a positive and collaborative atmosphere.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/decision-jungles.git
   cd decision-jungles
   ```
3. Install dependencies:
   ```bash
   # Development installation with all dependencies
   pip install -e ".[performance,profiling,dev]"
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

1. Implement your changes, following the coding standards outlined below
2. Add or update tests to cover your changes
3. Update documentation to reflect your changes
4. Run the test suite to ensure all tests pass
5. Submit a pull request

## Pull Request Process

1. Ensure your code follows the project's coding standards
2. Update the documentation if necessary
3. Add tests for new functionality
4. Make sure all tests pass
5. Submit a pull request with a clear title and description
6. Respond to any feedback or requested changes

## Development Environment

We recommend using a virtual environment for development:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[performance,profiling,dev]"

# For improved performance, compile Cython extensions
python setup_cython.py build_ext --inplace
```

## Coding Standards

We follow the general Python style guidelines:

- Follow [PEP 8](https://pep8.org/) for code style
- Add type hints to all functions and methods
- Document code using Google-style docstrings
- Keep functions focused and concise
- Write clear, descriptive variable and function names
- Maintain backward compatibility where possible

## Testing

All new features and bug fixes should include tests. We use pytest for testing:

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_specific_file.py

# Run tests with coverage
pytest --cov=decision_jungles tests/
```

Our test suite includes:
- Unit tests for individual components
- Integration tests for end-to-end functionality
- Property-based tests using Hypothesis
- Edge case tests

## Documentation

Good documentation is crucial. Please follow these guidelines:

- Update docstrings for any modified functions or classes
- Keep README.md and other documentation files up to date
- Add examples for new features
- Document parameters, return values, and exceptions
- Include references to papers or external resources when relevant

## Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check the issue tracker to see if it's already reported
2. If not, create a new issue with:
   - A clear title and description
   - Steps to reproduce (for bugs)
   - Expected behavior
   - Actual behavior
   - Environment details (OS, Python version, package versions)
   - Screenshots or code samples if applicable

## Feature Requests

We welcome feature requests! When proposing a new feature:

1. Describe the problem your feature would solve
2. Explain how your solution would work
3. Discuss alternatives you've considered
4. Note any potential challenges or concerns

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.