---
layout: default
title: Decision Jungles
nav_order: 1
---

# Decision Jungles

A scikit-learn compatible implementation of Decision Jungles as described in the paper ["Decision Jungles: Compact and Rich Models for Classification"](https://www.microsoft.com/en-us/research/publication/decision-jungles-compact-and-rich-models-for-classification/) by Jamie Shotton et al. (NIPS 2013).

## Overview

Decision Jungles are ensembles of rooted decision directed acyclic graphs (DAGs) that offer two key advantages over traditional decision trees/forests:

1. Reduced memory footprint through node merging
2. Improved generalization through regularization effects of the DAG structure

Unlike conventional decision trees that only allow one path to every node, a DAG in a decision jungle allows multiple paths from the root to each leaf. This results in a more compact model with potentially better generalization.

## Installation

### Requirements

- Python 3.8 or higher
- NumPy 1.17.0 or higher
- scikit-learn 0.21.0 or higher
- scipy 1.3.0 or higher

### Basic Installation

```bash
pip install decision-jungles
```

### Performance Optimization with Cython

```bash
# Install with Cython dependencies for improved performance
pip install decision-jungles[performance]

# Install with memory profiling and benchmarking tools
pip install decision-jungles[profiling]

# Install with development dependencies (testing, etc.)
pip install decision-jungles[dev]

# Install all optional dependencies
pip install decision-jungles[performance,profiling,dev]
```

## Quick Start

### Classification

```python
from decision_jungles import DecisionJungleClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Jungle
clf = DecisionJungleClassifier(
    n_estimators=10,
    max_width=256,
    max_depth=10,
    random_state=42
)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")

# Get memory usage
print(f"Memory usage: {clf.get_memory_usage()} bytes")
print(f"Number of nodes: {clf.get_node_count()}")
```

### Regression

```python
from decision_jungles import DecisionJungleRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Jungle for regression
reg = DecisionJungleRegressor(
    n_estimators=10,
    max_width=256,
    criterion="mse",  # Use "mse" or "mae"
    random_state=42
)
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print(f"R² score: {r2_score(y_test, y_pred):.4f}")

# Get memory usage
print(f"Memory usage: {reg.get_memory_usage()} bytes")
print(f"Number of nodes: {reg.get_node_count()}")
```

## Documentation

- [API Reference](api_reference.html)
- [User Guide](user_guide.html)
- [Examples](examples.html)
- [Regression Support](regression.html)

## Resources

- [GitHub Repository](https://github.com/example/decision-jungles)
- [PyPI Package](https://pypi.org/project/decision-jungles/)
- [Issue Tracker](https://github.com/example/decision-jungles/issues)

## Citation

If you use Decision Jungles in your research, please cite the original paper:

```bibtex
@inproceedings{shotton2013decision,
  title={Decision jungles: Compact and rich models for classification},
  author={Shotton, Jamie and Sharp, Toby and Kohli, Pushmeet and Nowozin, Sebastian and Winn, John and Criminisi, Antonio},
  booktitle={Advances in Neural Information Processing Systems},
  pages={234--242},
  year={2013}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](license.html) file for details.