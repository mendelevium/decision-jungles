---
layout: default
title: Regression Support
nav_order: 4
---

# Regression Support in Decision Jungles

This document provides detailed information about the regression capabilities of the Decision Jungles library.

## Introduction

Decision Jungles support regression tasks through the `DecisionJungleRegressor` class, which follows scikit-learn's estimator interface. This implementation extends the original classification-focused algorithm to regression tasks, allowing for prediction of continuous target values while maintaining the memory efficiency benefits of the DAG structure.

## Key Features

- **Memory Efficiency**: Like its classification counterpart, `DecisionJungleRegressor` maintains the memory efficiency benefits of node merging and DAG structures.
- **Multiple Impurity Criteria**: Supports both Mean Squared Error (MSE) and Mean Absolute Error (MAE) criteria for split quality evaluation.
- **Categorical Feature Support**: Automatic or manual handling of categorical features without preprocessing.
- **Early Stopping**: Optional early stopping functionality to prevent overfitting.
- **Scikit-learn Compatible**: Fully compatible with scikit-learn's cross-validation, pipelines, and hyperparameter tuning tools.

## Usage

### Basic Usage

```python
from decision_jungles import DecisionJungleRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load a regression dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a DecisionJungleRegressor
regressor = DecisionJungleRegressor(
    n_estimators=20,      # Number of DAGs in the ensemble
    max_width=128,        # Maximum width of each level in the DAGs
    criterion="mse",      # Split quality criterion (mse or mae)
    random_state=42
)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate performance
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
```

### Handling Categorical Features

```python
import numpy as np
from decision_jungles import DecisionJungleRegressor

# Create dataset with categorical features
X = np.random.randn(200, 5)
X[:, 0] = np.random.randint(0, 3, size=200)  # Categorical feature with 3 values
X[:, 1] = np.random.randint(0, 5, size=200)  # Categorical feature with 5 values

# Create target based on categorical features
y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + np.random.randn(200) * 0.5

# Approach 1: Explicitly specify categorical features
regressor = DecisionJungleRegressor(
    categorical_features=[0, 1],  # Indices of categorical features
    max_bins=10                   # Maximum number of categories per feature
)

# Approach 2: Automatic detection of categorical features
regressor = DecisionJungleRegressor(
    categorical_features="auto",  # Automatically detect categorical features
    max_bins=10                   # Maximum number of categories per feature
)
```

### Early Stopping

```python
from decision_jungles import DecisionJungleRegressor

regressor = DecisionJungleRegressor(
    early_stopping=True,          # Enable early stopping
    validation_fraction=0.1,      # Fraction of training data to use for validation
    n_iter_no_change=5,           # Number of iterations with no improvement to wait
    tol=1e-4                      # Minimum improvement required to continue
)
```

## Parameters

The `DecisionJungleRegressor` supports the following parameters:

- **n_estimators** (int, default=10): Number of DAGs in the jungle.
- **max_width** (int, default=256): Maximum width of each level in the DAGs.
- **max_depth** (int, default=None): Maximum depth of the DAGs.
- **min_samples_split** (int, default=2): Minimum samples required to split a node.
- **min_samples_leaf** (int, default=1): Minimum samples required in a leaf node.
- **min_impurity_decrease** (float, default=0.0): Minimum decrease in impurity required for a split.
- **max_features** (int, float, str, default="auto"): Number of features to consider for splits.
- **criterion** (str, default="mse"): Function to measure split quality ("mse" or "mae").
- **random_state** (int, default=None): Seed for random number generation.
- **merging_schedule** (str, default="exponential"): Controls how width increases with depth.
- **optimization_method** (str, default="lsearch"): Algorithm for DAG optimization.
- **n_jobs** (int, default=None): Number of parallel jobs.
- **use_optimized** (bool, default=True): Whether to use optimized algorithm implementations.
- **early_stopping** (bool, default=False): Whether to use early stopping.
- **validation_fraction** (float, default=0.1): Fraction of training data for validation.
- **n_iter_no_change** (int, default=5): Number of iterations with no improvement before stopping.
- **tol** (float, default=1e-4): Tolerance for improvement detection.
- **categorical_features** (array-like or str, default=None): Specifies categorical features.
- **max_bins** (int, default=255): Maximum number of bins for categorical features.

## Attributes

After fitting, the following attributes are available:

- **dags_**: List of fitted RegressionDAG objects.
- **n_features_in_**: Number of features seen during fit.
- **feature_importances_**: Importance of each feature.

## Methods

The regressor provides the following methods:

- **fit(X, y)**: Fit the model to training data.
- **predict(X)**: Predict target values for samples in X.
- **get_memory_usage()**: Calculate total memory usage in bytes.
- **get_node_count()**: Get total number of nodes in the ensemble.
- **get_max_depth()**: Get maximum depth across all DAGs.

## Implementation Details

The implementation of regression support introduces several new components:

1. **RegressionLeafNode**: A specialized leaf node that stores and predicts continuous values instead of class probabilities.
2. **RegressionDAG**: A modified DAG structure optimized for regression tasks.
3. **Regression-specific objective functions**: MSE and MAE impurity measures for evaluating split quality.

## Performance Comparison

When compared to Random Forests for regression, Decision Jungles typically show:

- Significantly reduced memory usage (often 50-80% less)
- Comparable prediction accuracy (R² scores within 5-10% of Random Forests)
- Slightly longer training times
- Similar prediction times

## Limitations

- Training time can be longer than traditional random forests due to the optimization process
- The optimization methods (LSearch, ClusterSearch) were originally designed for classification tasks and may not be optimal for all regression scenarios