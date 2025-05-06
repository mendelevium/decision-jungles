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

For better performance, you can compile the Cython extensions:

```bash
# Install Cython
pip install cython

# Clone the repository (if installing from source)
git clone https://github.com/mendelevium/decision-jungles.git
cd decision-jungles

# Compile Cython extensions
python setup_cython.py build_ext --inplace
```

This will significantly speed up the training process, especially for large datasets.

## Usage

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

## Key Features

- Scikit-learn compatible API for both classification and regression
- Two node merging algorithms: LSearch and ClusterSearch
- Various merging schedules for different applications
- Memory-efficient implementation with significant space savings compared to Random Forests
- Support for both classification (gini, entropy) and regression (MSE, MAE) criteria
- Visualization utilities for model inspection and interpretation
- Performance metrics and memory profiling tools
- Robust model serialization with pickle and joblib
- Direct support for categorical features without preprocessing
- Feature importance calculation
- Early stopping functionality to prevent overfitting

## Parameters

### Classification

The `DecisionJungleClassifier` accepts the following parameters:

- `n_estimators` (int, default=10): Number of DAGs in the jungle.
- `max_width` (int, default=256): Maximum width of each level (M parameter in the paper).
- `max_depth` (int, default=None): Maximum depth of the DAGs.
- `min_samples_split` (int, default=2): Minimum number of samples required to split a node.
- `min_samples_leaf` (int, default=1): Minimum number of samples required at a leaf node.
- `min_impurity_decrease` (float, default=0.0): Minimum impurity decrease required for a split.
- `max_features` (int, float, str, default="sqrt"): Number of features to consider for best split.
- `random_state` (int, default=None): Random seed for reproducibility.
- `merging_schedule` (str, default="exponential"): Type of merging schedule to use.
- `n_jobs` (int, default=None): Number of jobs to run in parallel.
- `categorical_features` (array-like or str, default=None): Specifies which features are categorical.
- `early_stopping` (bool, default=False): Whether to use early stopping during training.

### Regression

The `DecisionJungleRegressor` accepts similar parameters with these differences:

- `criterion` (str, default="mse"): Function to measure split quality:
  - "mse": Mean squared error minimization
  - "mae": Mean absolute error minimization
- `max_features` (int, float, str, default="auto"): Number of features to consider for best split.

## Comparison with Decision Forests

Decision Jungles offer several advantages over traditional Decision Forests:

1. **Memory Efficiency**: Jungles require dramatically less memory while often improving generalization.
2. **Improved Generalization**: Node merging can lead to better regularization and improved test accuracy.
3. **Inference Time**: For the same memory footprint, jungles can achieve higher accuracy than forests.

## Examples

Check the `examples/` directory for various usage examples:

- Basic usage
- Comparison with scikit-learn's Random Forests
- Memory usage analysis
- Performance on different datasets
- Hyperparameter tuning
- Categorical features handling
- Model serialization and loading
- Early stopping for preventing overfitting
- Integration with scikit-learn pipelines

### Model Serialization

Decision Jungle models can be easily saved and loaded using standard Python serialization mechanisms:

```python
import pickle
import joblib
from decision_jungles import DecisionJungleClassifier

# Train a model
jungle = DecisionJungleClassifier(n_estimators=10)
jungle.fit(X_train, y_train)

# Method 1: Save model using pickle
with open("jungle_model.pkl", "wb") as f:
    pickle.dump(jungle, f)

# Method 1: Load model using pickle
with open("jungle_model.pkl", "rb") as f:
    loaded_jungle = pickle.load(f)

# Method 2: Save model using joblib (better for large models)
joblib.dump(jungle, "jungle_model.joblib")

# Method 2: Load model using joblib
loaded_jungle = joblib.load("jungle_model.joblib")

# Make predictions with the loaded model
predictions = loaded_jungle.predict(X_test)
```

## Documentation

### Regression

For detailed information about the regression functionality, please see the [README_REGRESSION.md](README_REGRESSION.md) file, which includes:

- Detailed implementation overview
- Advanced usage examples
- Performance comparisons with Random Forest Regressors
- Implementation details and architecture

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Decision Jungles in your research, please cite the original paper:

```
@inproceedings{shotton2013decision,
  title={Decision jungles: Compact and rich models for classification},
  author={Shotton, Jamie and Sharp, Toby and Kohli, Pushmeet and Nowozin, Sebastian and Winn, John and Criminisi, Antonio},
  booktitle={Advances in Neural Information Processing Systems},
  pages={234--242},
  year={2013}
}
```
