"""
Tests for edge cases in Decision Jungles.
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from decision_jungles import DecisionJungleClassifier
from decision_jungles.dag import DAG


def test_empty_dataset():
    """Test behavior with an empty dataset."""
    # Create an empty dataset
    X = np.array([]).reshape(0, 5)
    y = np.array([])
    
    # Check that fitting raises an appropriate error
    with pytest.raises(ValueError):
        clf = DecisionJungleClassifier(random_state=42)
        clf.fit(X, y)


def test_single_sample():
    """Test behavior with a single sample."""
    # Create a dataset with a single sample
    X = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
    y = np.array([0])
    
    # Should be able to fit and predict
    clf = DecisionJungleClassifier(random_state=42)
    clf.fit(X, y)
    
    # Prediction should return the only class seen
    assert clf.predict(X)[0] == 0
    
    # Check probability (should be 1.0 for the only class)
    proba = clf.predict_proba(X)
    assert proba.shape == (1, 1)
    assert proba[0, 0] == 1.0


def test_single_class():
    """Test behavior when all samples belong to a single class."""
    # Create a dataset with multiple samples but only one class
    X = np.random.rand(10, 5)
    y = np.zeros(10, dtype=int)
    
    # Should fit without errors
    clf = DecisionJungleClassifier(random_state=42)
    clf.fit(X, y)
    
    # All predictions should be the same class
    preds = clf.predict(X)
    assert np.all(preds == 0)
    
    # Probabilities should all be 1.0 for the single class
    proba = clf.predict_proba(X)
    assert proba.shape == (10, 1)
    assert np.all(proba == 1.0)


def test_missing_values():
    """Test behavior with missing values (NaN)."""
    # Create a dataset with missing values
    X = np.random.rand(20, 5)
    X[3, 2] = np.nan  # Add a NaN value
    X[10, 4] = np.nan  # Add another NaN value
    y = np.random.randint(0, 2, 20)
    
    # Should handle missing values gracefully (NaN are sent to the right child)
    # This test previously expected an error, but now the code handles NaN values
    clf = DecisionJungleClassifier(random_state=42)
    clf.fit(X, y)
    
    # Make sure we can predict with data containing NaN
    X_test = np.random.rand(5, 5)
    X_test[2, 3] = np.nan
    y_pred = clf.predict(X_test)
    assert len(y_pred) == 5


def test_predict_before_fit():
    """Test behavior when trying to predict before fitting."""
    # Create a classifier without fitting
    clf = DecisionJungleClassifier()
    
    # Create some test data
    X = np.random.rand(5, 3)
    
    # Should raise a NotFittedError
    with pytest.raises(NotFittedError):
        clf.predict(X)
    
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)


def test_high_dimensional_data():
    """Test with high-dimensional data."""
    # Create a high-dimensional dataset (100 features)
    X = np.random.rand(30, 100)
    y = np.random.randint(0, 3, 30)
    
    # Should fit and predict without errors
    clf = DecisionJungleClassifier(random_state=42)
    clf.fit(X, y)
    
    # Make predictions
    preds = clf.predict(X)
    assert preds.shape == (30,)
    
    # Check probability predictions
    proba = clf.predict_proba(X)
    assert proba.shape == (30, 3)
    assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_inconsistent_features():
    """Test behavior when number of features in training and testing differ."""
    # Train with 5 features
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, 10)
    
    # Test with 3 features (inconsistent)
    X_test = np.random.rand(5, 3)
    
    clf = DecisionJungleClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Should raise an appropriate error
    with pytest.raises(ValueError):
        clf.predict(X_test)