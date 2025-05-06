"""
Tests for the DecisionJungleClassifier.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_jungles.jungle import DecisionJungleClassifier


def test_jungle_initialization():
    """Test initialization of DecisionJungleClassifier."""
    jungle = DecisionJungleClassifier(
        n_estimators=5,
        max_width=128,
        max_depth=10,
        random_state=42
    )
    
    assert jungle.n_estimators == 5
    assert jungle.max_width == 128
    assert jungle.max_depth == 10
    assert jungle.random_state == 42


def test_parameter_validation():
    """Test validation of parameters."""
    # Invalid n_estimators
    with pytest.raises(ValueError):
        DecisionJungleClassifier(n_estimators=0)
    
    # Invalid max_width
    with pytest.raises(ValueError):
        DecisionJungleClassifier(max_width=0)
    
    # Invalid max_depth
    with pytest.raises(ValueError):
        DecisionJungleClassifier(max_depth=0)
    
    # Invalid min_samples_split
    with pytest.raises(ValueError):
        DecisionJungleClassifier(min_samples_split=1)
    
    # Invalid min_samples_leaf
    with pytest.raises(ValueError):
        DecisionJungleClassifier(min_samples_leaf=0)
    
    # Invalid min_impurity_decrease
    with pytest.raises(ValueError):
        DecisionJungleClassifier(min_impurity_decrease=-0.1)
    
    # Invalid max_features
    with pytest.raises(ValueError):
        DecisionJungleClassifier(max_features="invalid")
    
    # Invalid merging_schedule
    with pytest.raises(ValueError):
        DecisionJungleClassifier(merging_schedule="invalid")


def test_fit_iris():
    """Test fitting on the Iris dataset."""
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit a small jungle
    jungle = DecisionJungleClassifier(
        n_estimators=3,
        max_width=16,
        max_depth=5,
        random_state=42
    )
    jungle.fit(X_train, y_train)
    
    # Check attributes
    assert hasattr(jungle, 'dags_')
    assert len(jungle.dags_) == 3
    assert hasattr(jungle, 'classes_')
    assert len(jungle.classes_) == 3
    assert hasattr(jungle, 'n_classes_')
    assert jungle.n_classes_ == 3
    
    # Predict and check accuracy
    y_pred = jungle.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Accuracy should be reasonable for Iris
    assert accuracy > 0.8


def test_predict_proba():
    """Test probability predictions."""
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit a small jungle
    jungle = DecisionJungleClassifier(
        n_estimators=3,
        max_width=16,
        max_depth=5,
        random_state=42
    )
    jungle.fit(X_train, y_train)
    
    # Get probability predictions
    proba = jungle.predict_proba(X_test)
    
    # Check shape and properties
    assert proba.shape == (len(X_test), 3)  # 3 classes
    assert np.all(proba >= 0)  # Probabilities should be non-negative
    assert np.all(proba <= 1)  # Probabilities should be at most 1
    assert np.allclose(np.sum(proba, axis=1), 1.0)  # Probabilities should sum to 1
    
    # Check predictions based on probabilities
    pred_from_proba = np.argmax(proba, axis=1)
    pred_direct = jungle.predict(X_test)
    assert np.array_equal(pred_from_proba, pred_direct)


def test_binary_classification():
    """Test binary classification."""
    # Load breast cancer dataset (binary classification)
    X, y = load_breast_cancer(return_X_y=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit a small jungle
    jungle = DecisionJungleClassifier(
        n_estimators=3,
        max_width=16,
        max_depth=5,
        random_state=42
    )
    jungle.fit(X_train, y_train)
    
    # Check attributes
    assert hasattr(jungle, 'dags_')
    assert len(jungle.dags_) == 3
    assert hasattr(jungle, 'classes_')
    assert len(jungle.classes_) == 2
    assert hasattr(jungle, 'n_classes_')
    assert jungle.n_classes_ == 2
    
    # Predict and check accuracy
    y_pred = jungle.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Accuracy should be reasonable for breast cancer
    assert accuracy > 0.8


def test_memory_usage():
    """Test memory usage calculation."""
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    
    # Train small and large jungles
    jungle_small = DecisionJungleClassifier(
        n_estimators=2,
        max_width=8,
        max_depth=3,
        random_state=42
    )
    jungle_small.fit(X, y)
    
    jungle_large = DecisionJungleClassifier(
        n_estimators=5,
        max_width=32,
        max_depth=5,
        random_state=42
    )
    jungle_large.fit(X, y)
    
    # Calculate memory usage
    memory_small = jungle_small.get_memory_usage()
    memory_large = jungle_large.get_memory_usage()
    
    # The larger jungle should use more memory
    assert memory_large > memory_small
    
    # Memory usage should match the sum of DAG memory usage
    assert memory_small == sum(dag.get_memory_usage() for dag in jungle_small.dags_)


def test_node_count():
    """Test node counting."""
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    
    # Train small and large jungles
    jungle_small = DecisionJungleClassifier(
        n_estimators=2,
        max_width=8,
        max_depth=3,
        random_state=42
    )
    jungle_small.fit(X, y)
    
    jungle_large = DecisionJungleClassifier(
        n_estimators=5,
        max_width=32,
        max_depth=5,
        random_state=42
    )
    jungle_large.fit(X, y)
    
    # Calculate node counts
    nodes_small = jungle_small.get_node_count()
    nodes_large = jungle_large.get_node_count()
    
    # The larger jungle should have more nodes
    assert nodes_large > nodes_small
    
    # Node count should match the sum of DAG node counts
    assert nodes_small == sum(dag.get_node_count() for dag in jungle_small.dags_)


def test_max_depth():
    """Test max depth calculation."""
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    
    # Train jungle with limited depth
    jungle = DecisionJungleClassifier(
        n_estimators=3,
        max_width=32,
        max_depth=4,
        random_state=42
    )
    jungle.fit(X, y)
    
    # Get max depth
    max_depth = jungle.get_max_depth()
    
    # Max depth should not exceed limit
    assert max_depth <= 4
    
    # Max depth should match the max of DAG max depths
    assert max_depth == max(dag.get_max_depth() for dag in jungle.dags_)
