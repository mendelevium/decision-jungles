# Decision Jungles API Reference

This document provides a comprehensive reference for the Decision Jungles library, listing all the public classes, methods, and attributes with their descriptions and usage examples.

## Table of Contents

- [DecisionJungleClassifier](#decisionjungleclassifier)
- [DAG](#dag)
- [Node Classes](#node-classes)
  - [SplitNode](#splitnode)
  - [LeafNode](#leafnode)
- [Training Algorithms](#training-algorithms)
  - [LSearch](#lsearch)
  - [OptimizedLSearch](#optimizedlsearch)
  - [ClusterSearch](#clustersearch)
- [Utilities](#utilities)
  - [Memory Profiling](#memory-profiling)
  - [Benchmarking](#benchmarking)
  - [Visualization](#visualization)
  - [Metrics](#metrics)

## DecisionJungleClassifier

The main classifier class implementing scikit-learn's estimator interface.

```python
from decision_jungles import DecisionJungleClassifier
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_estimators | int | 10 | Number of DAGs in the ensemble |
| max_width | int | 256 | Maximum width of each level (M parameter) |
| max_depth | int, None | None | Maximum depth of the DAGs |
| min_samples_split | int | 2 | Minimum samples required to split a node |
| min_samples_leaf | int | 1 | Minimum samples required at a leaf node |
| min_impurity_decrease | float | 0.0 | Minimum impurity decrease for a split |
| max_features | int, float, str | "sqrt" | Number of features to consider for best split |
| random_state | int, None | None | Random seed for reproducibility |
| merging_schedule | str | "exponential" | Type of merging schedule ("constant", "exponential", "kinect") |
| optimization_method | str | "lsearch" | Method for DAG optimization ("lsearch" or "clustersearch") |
| n_jobs | int, None | None | Number of jobs for parallel processing |
| use_optimized | bool | True | Whether to use the optimized implementation of LSearch |
| early_stopping | bool | False | Whether to use early stopping to terminate training |
| validation_fraction | float | 0.1 | Proportion of training data to use for validation in early stopping |
| n_iter_no_change | int | 5 | Number of iterations with no improvement before early stopping |
| tol | float | 1e-4 | Tolerance for improvement in validation score |
| categorical_features | None, "auto", list, array | None | Specifies which features to treat as categorical. None: no categorical features. "auto": automatically detect. Boolean array: boolean mask. Integer array: feature indices. String array: feature names. |
| max_bins | int | 255 | Maximum number of bins for categorical features |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| n_classes_ | int | Number of classes in the classification problem |
| classes_ | ndarray | Class labels |
| dags_ | list | Collection of fitted DAGs |
| n_features_in_ | int | Number of features seen during fit |
| feature_importances_ | ndarray | Importance of each feature based on frequency of use in split nodes |
| stopped_early_ | bool | Whether early stopping was triggered during training (only if early_stopping=True) |
| is_categorical_ | ndarray | Boolean mask indicating which features are treated as categorical |
| feature_bins_ | Dict | Dictionary mapping feature indices to category-to-bin mappings |

### Methods

#### fit

```python
def fit(X, y)
```

Build a decision jungle ensemble from the training data.

**Parameters:**
- X: array-like of shape (n_samples, n_features) - The training input samples
- y: array-like of shape (n_samples,) - The target values (class labels)

**Returns:**
- self: DecisionJungleClassifier - The fitted estimator

#### predict_proba

```python
def predict_proba(X)
```

Predict class probabilities for the input samples.

**Parameters:**
- X: array-like of shape (n_samples, n_features) - The input samples

**Returns:**
- proba: ndarray of shape (n_samples, n_classes) - Class probabilities for each sample

#### predict

```python
def predict(X)
```

Predict class labels for the input samples.

**Parameters:**
- X: array-like of shape (n_samples, n_features) - The input samples

**Returns:**
- y: ndarray of shape (n_samples,) - The predicted classes

#### get_memory_usage

```python
def get_memory_usage()
```

Calculate the total memory usage of the jungle in bytes.

**Returns:**
- int: Total memory usage in bytes

#### get_node_count

```python
def get_node_count()
```

Get the total number of nodes in the jungle.

**Returns:**
- int: Total number of nodes across all DAGs

#### get_max_depth

```python
def get_max_depth()
```

Get the maximum depth across all DAGs in the jungle.

**Returns:**
- int: Maximum depth

#### feature_importances_

```python
@property
def feature_importances_
```

The feature importances based on feature usage in the jungle.

The importance of a feature is calculated as the normalized count of how many times that feature is used for splitting across all nodes in all DAGs of the ensemble.

**Returns:**
- ndarray of shape (n_features,): Normalized array of feature importances. Higher values indicate more important features.

**Raises:**
- NotFittedError: If the model has not been fitted yet.

#### __getstate__

```python
def __getstate__()
```

Return the state of the estimator for pickling.

This method is called when pickling the estimator. It returns the internal state of the object, which can then be used to restore the object when unpickling.

**Returns:**
- dict: The state of the estimator to be serialized.

#### __setstate__

```python
def __setstate__(state)
```

Set the state of the estimator after unpickling.

This method is called when unpickling the estimator. It restores the internal state of the object.

**Parameters:**
- state: dict - The state of the estimator as returned by __getstate__().

### Usage Examples

#### Basic Example

```python
from decision_jungles import DecisionJungleClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create and train model
jungle = DecisionJungleClassifier(
    n_estimators=10,
    max_width=64,
    max_depth=10,
    random_state=42
)
jungle.fit(X_train, y_train)

# Make predictions
y_pred = jungle.predict(X_test)
probas = jungle.predict_proba(X_test)

# Get memory usage info
memory_bytes = jungle.get_memory_usage()
node_count = jungle.get_node_count()

# Get feature importances
importances = jungle.feature_importances_
feature_names = iris.feature_names
for i, importance in enumerate(importances):
    print(f"Feature {feature_names[i]}: {importance:.4f}")
```

#### Categorical Features Example

```python
import pandas as pd
import numpy as np
from decision_jungles import DecisionJungleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data with categorical features (e.g., UCI Car Evaluation dataset)
# For demonstration, we'll create a simple synthetic dataset with categorical features
np.random.seed(42)

# Create synthetic data with categorical features
n_samples = 1000
# Numerical features
numerical_feat1 = np.random.normal(0, 1, n_samples)
numerical_feat2 = np.random.normal(0, 1, n_samples)

# Categorical features (3 categories each)
cat_feat1 = np.random.randint(0, 3, n_samples)  # Categories: 0, 1, 2
cat_feat2 = np.random.randint(0, 3, n_samples)  # Categories: 0, 1, 2

# Target depends on both numerical and categorical features
y = ((numerical_feat1 > 0) & (cat_feat1 == 2)) | ((numerical_feat2 < 0) & (cat_feat2 == 1))
y = y.astype(int)

# Create a DataFrame (easier to work with categorical data)
X = pd.DataFrame({
    'num1': numerical_feat1,
    'num2': numerical_feat2,
    'cat1': cat_feat1,
    'cat2': cat_feat2
})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to numpy for consistency
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

# 1. Using automatic categorical feature detection
dj_auto = DecisionJungleClassifier(
    n_estimators=50,
    categorical_features="auto",  # Automatically detect categorical features
    max_bins=10,
    random_state=42
)
dj_auto.fit(X_train_np, y_train)
y_pred_auto = dj_auto.predict(X_test_np)
acc_auto = accuracy_score(y_test, y_pred_auto)
print(f"Decision Jungle (auto categorical): {acc_auto:.4f}")

# 2. Using explicit categorical feature indices
cat_indices = [2, 3]  # Indices of categorical features (cat1, cat2)
dj_explicit = DecisionJungleClassifier(
    n_estimators=50,
    categorical_features=cat_indices,  # Explicitly specify categorical features
    max_bins=10,
    random_state=42
)
dj_explicit.fit(X_train_np, y_train)
y_pred_explicit = dj_explicit.predict(X_test_np)
acc_explicit = accuracy_score(y_test, y_pred_explicit)
print(f"Decision Jungle (explicit categorical): {acc_explicit:.4f}")

# 3. Using boolean mask for categorical features
cat_mask = np.array([False, False, True, True])  # Boolean mask (num1, num2, cat1, cat2)
dj_mask = DecisionJungleClassifier(
    n_estimators=50,
    categorical_features=cat_mask,  # Use boolean mask
    max_bins=10,
    random_state=42
)
dj_mask.fit(X_train_np, y_train)
y_pred_mask = dj_mask.predict(X_test_np)
acc_mask = accuracy_score(y_test, y_pred_mask)
print(f"Decision Jungle (boolean mask): {acc_mask:.4f}")

# Compare with standard Decision Jungle (no categorical features)
dj_standard = DecisionJungleClassifier(
    n_estimators=50,
    random_state=42
)
dj_standard.fit(X_train_np, y_train)
y_pred_std = dj_standard.predict(X_test_np)
acc_std = accuracy_score(y_test, y_pred_std)
print(f"Decision Jungle (no categorical): {acc_std:.4f}")
```

## DAG

A Directed Acyclic Graph (DAG) for Decision Jungle classification.

```python
from decision_jungles.dag import DAG
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_classes | int | - | Number of classes in the classification problem |
| max_depth | int, None | None | Maximum depth of the DAG |
| min_samples_split | int | 2 | Minimum samples required to split a node |
| min_samples_leaf | int | 1 | Minimum samples required at a leaf node |
| min_impurity_decrease | float | 0.0 | Minimum impurity decrease for a split |
| max_features | int, float, str | "sqrt" | Number of features to consider |
| random_state | int, None | None | Random seed for reproducibility |
| merging_schedule | str | "exponential" | Type of merging schedule |
| max_width | int | 256 | Maximum width of each level |
| optimization_method | str | "lsearch" | Method for optimization |
| use_optimized | bool | True | Whether to use optimized implementation |
| early_stopping | bool | False | Whether to use early stopping |
| validation_X | ndarray, None | None | Validation features for early stopping |
| validation_y | ndarray, None | None | Validation labels for early stopping |
| n_iter_no_change | int | 5 | Iterations with no improvement before early stopping |
| tol | float | 1e-4 | Tolerance for early stopping criterion |
| is_categorical | ndarray, None | None | Boolean mask indicating categorical features |
| feature_bins | Dict, None | None | Mapping of feature indices to category bins |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| nodes | Dict[int, Node] | Dictionary of nodes keyed by node ID |
| root_node_id | int | ID of the root node |
| n_classes | int | Number of classes in the classification problem |
| next_node_id | int | Counter for generating unique node IDs |

### Methods

#### fit

```python
def fit(X, y)
```

Fit the DAG to the training data.

**Parameters:**
- X: ndarray of shape (n_samples, n_features) - The training input samples
- y: ndarray of shape (n_samples,) - The target values (class labels)

#### predict_proba

```python
def predict_proba(X)
```

Predict class probabilities for the input samples.

**Parameters:**
- X: ndarray of shape (n_samples, n_features) - The input samples

**Returns:**
- proba: ndarray of shape (n_samples, n_classes) - Class probabilities for each sample

#### predict

```python
def predict(X)
```

Predict class labels for the input samples.

**Parameters:**
- X: ndarray of shape (n_samples, n_features) - The input samples

**Returns:**
- y: ndarray of shape (n_samples,) - The predicted classes

#### get_memory_usage

```python
def get_memory_usage()
```

Calculate the memory usage of the DAG in bytes.

**Returns:**
- int: Memory usage in bytes

#### get_node_count

```python
def get_node_count()
```

Get the number of nodes in the DAG.

**Returns:**
- int: Total number of nodes

#### get_max_depth

```python
def get_max_depth()
```

Get the maximum depth of the DAG.

**Returns:**
- int: Maximum depth of any node in the DAG

#### __getstate__

```python
def __getstate__()
```

Return the state of the DAG for pickling.

This method is called when pickling the DAG. It returns the internal state of the object, which can then be used to restore the object when unpickling.

**Returns:**
- dict: The state of the DAG to be serialized.

#### __setstate__

```python
def __setstate__(state)
```

Set the state of the DAG after unpickling.

This method is called when unpickling the DAG. It restores the internal state of the object.

**Parameters:**
- state: dict - The state of the DAG as returned by __getstate__().

## Node Classes

### SplitNode

Internal node with splitting functionality.

```python
from decision_jungles.node import SplitNode
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| node_id | int | Unique identifier for the node |
| depth | int | Depth of the node in the DAG |
| feature_idx | int | Index of the feature used for splitting |
| threshold | float | Threshold value for the split (for numerical features) |
| is_categorical | bool | Whether this node splits on a categorical feature |
| categories_left | Set, None | For categorical features, the set of category values that go to the left child |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| node_id | int | Unique identifier for the node |
| depth | int | Depth of the node in the DAG |
| feature_idx | int | Index of the feature used for splitting |
| threshold | float | Threshold value for the split |
| left_child | int | ID of the left child node |
| right_child | int | ID of the right child node |
| parents | Set[int] | Set of parent node IDs |
| is_categorical | bool | Whether this node splits on a categorical feature |
| categories_left | Set | For categorical features, the set of category values that go to the left child |

#### Methods

##### set_children

```python
def set_children(left_child, right_child)
```

Set the child nodes of this split node.

##### add_parent

```python
def add_parent(parent_id)
```

Add a parent node to this node.

##### predict

```python
def predict(X)
```

Predict which branch (left/right) samples should follow.

##### get_memory_usage

```python
def get_memory_usage()
```

Calculate the memory usage of this node in bytes.

### LeafNode

Terminal node with class distribution.

```python
from decision_jungles.node import LeafNode
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| node_id | int | Unique identifier for the node |
| depth | int | Depth of the node in the DAG |
| n_classes | int | Number of classes in the problem |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| node_id | int | Unique identifier for the node |
| depth | int | Depth of the node in the DAG |
| n_classes | int | Number of classes in the problem |
| class_distribution | ndarray | Array of class weights |
| n_samples | int | Number of samples in this node |
| parents | Set[int] | Set of parent node IDs |

#### Methods

##### add_parent

```python
def add_parent(parent_id)
```

Add a parent node to this node.

##### update_distribution

```python
def update_distribution(y)
```

Update the class distribution based on training samples.

##### predict

```python
def predict(X)
```

Predict class probabilities for the input samples.

##### get_memory_usage

```python
def get_memory_usage()
```

Calculate the memory usage of this node in bytes.

## Training Algorithms

### LSearch

Implementation of the LSearch optimization algorithm for Decision Jungles.

```python
from decision_jungles.training.lsearch import LSearch
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_classes | int | - | Number of classes in the dataset |
| max_features | int, float, str | "sqrt" | Features to consider for each split |
| random_state | int, None | None | Random seed for reproducibility |

#### Methods

##### optimize

```python
def optimize(X, y, parent_nodes, node_samples, n_child_nodes, max_iterations=100, is_categorical=None, feature_bins=None)
```

Optimize one level of the DAG using the LSearch algorithm. Supports categorical features.

### OptimizedLSearch

Optimized implementation of the LSearch algorithm using NumPy vectorization.

```python
from decision_jungles.training.optimized_lsearch import OptimizedLSearch
```

Similar interface to LSearch, but with improved performance through vectorization.

### ClusterSearch

Alternative optimization method using clustering for Decision Jungles.

```python
from decision_jungles.training.clustersearch import ClusterSearch
```

## Utilities

### Memory Profiling

Utilities for measuring and comparing memory usage.

```python
from decision_jungles.utils.memory_profiling import measure_model_memory, memory_usage_vs_accuracy, estimate_model_size
```

#### measure_model_memory

```python
def measure_model_memory(model, detailed=False)
```

Measure memory usage of a machine learning model.

#### memory_usage_vs_accuracy

```python
def memory_usage_vs_accuracy(jungle_constructor, forest_constructor, X_train, y_train, X_test, y_test, param_range, param_name="n_estimators", title="Memory Usage vs. Accuracy", save_path=None)
```

Compare memory usage vs. accuracy for Decision Jungles and Random Forests.

#### estimate_model_size

```python
def estimate_model_size(jungle_kwargs, average_nodes_per_dag=None)
```

Estimate memory usage of a Decision Jungle without creating it.

### Benchmarking

Tools for benchmarking performance.

```python
from decision_jungles.utils.benchmarking import time_execution, benchmark_optimization, benchmark_scaling, benchmark_hyperparameters
```

#### time_execution

```python
def time_execution(func, *args, **kwargs)
```

Measure execution time of a function.

#### benchmark_optimization

```python
def benchmark_optimization(X, y, jungle_class, n_repeats=3, n_estimators=5, max_width=64, max_depth=8, random_state=42)
```

Benchmark standard vs optimized implementations.

#### benchmark_scaling

```python
def benchmark_scaling(jungle_class, sizes=[1000, 10000, 50000], features=[10, 50, 100], n_estimators=5, max_width=64, max_depth=8, use_optimized=True, random_state=42, plot=True, save_path=None)
```

Benchmark scaling behavior with dataset size and dimensionality.

#### benchmark_hyperparameters

```python
def benchmark_hyperparameters(X, y, jungle_class, n_estimators_range=[1, 5, 10, 20], max_width_range=[16, 64, 256], max_depth_range=[5, 10, None], use_optimized=True, random_state=42, plot=True, save_path=None)
```

Benchmark performance across different hyperparameter settings.

### Visualization

Utilities for visualizing Decision Jungles.

```python
from decision_jungles.utils.visualization import plot_dag
```

#### plot_dag

```python
def plot_dag(dag, max_nodes=None, figsize=(12, 8), node_size=1000, font_size=10, save_path=None)
```

Plot a visualization of a DAG structure.

### Metrics

Metrics for evaluating Decision Jungles.

```python
from decision_jungles.utils.metrics import compare_memory_usage, measure_prediction_time
```

#### compare_memory_usage

```python
def compare_memory_usage(jungle, forest)
```

Compare memory usage between a Decision Jungle and a Random Forest.

#### measure_prediction_time

```python
def measure_prediction_time(model, X, n_repeats=10)
```

Measure prediction time of a model.