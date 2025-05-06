"""
Tests for the feature importance calculation in DecisionJungleClassifier.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose

from decision_jungles import DecisionJungleClassifier


def test_feature_importances_iris():
    """Test feature importances on the Iris dataset."""
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Fit the classifier
    clf = DecisionJungleClassifier(n_estimators=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Check if feature_importances_ is available
    assert hasattr(clf, 'feature_importances_')
    
    # Check dimensions
    assert clf.feature_importances_.shape == (X.shape[1],)
    
    # Check that importances sum to 1
    assert_allclose(np.sum(clf.feature_importances_), 1.0, rtol=1e-6)
    
    # Check if all values are non-negative
    assert np.all(clf.feature_importances_ >= 0)


def test_feature_importances_breast_cancer():
    """Test feature importances on a larger dataset (Breast Cancer)."""
    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Scale features to have similar ranges
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Fit the classifier
    clf = DecisionJungleClassifier(n_estimators=5, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Check if feature_importances_ is available
    assert hasattr(clf, 'feature_importances_')
    
    # Check dimensions
    assert clf.feature_importances_.shape == (X.shape[1],)
    
    # Check that importances sum to 1
    assert_allclose(np.sum(clf.feature_importances_), 1.0, rtol=1e-6)
    
    # Check if all values are non-negative
    assert np.all(clf.feature_importances_ >= 0)


def test_feature_importances_consistency():
    """Test that feature importances are consistent for the same random state."""
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Fit two identical classifiers
    clf1 = DecisionJungleClassifier(n_estimators=5, random_state=42)
    clf1.fit(X_train, y_train)
    
    clf2 = DecisionJungleClassifier(n_estimators=5, random_state=42)
    clf2.fit(X_train, y_train)
    
    # Check that importances are identical
    assert_array_equal(clf1.feature_importances_, clf2.feature_importances_)


def test_feature_importances_vary_with_random_state():
    """Test that feature importances vary with different random states."""
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Fit two classifiers with different random states
    clf1 = DecisionJungleClassifier(n_estimators=5, random_state=42)
    clf1.fit(X_train, y_train)
    
    clf2 = DecisionJungleClassifier(n_estimators=5, random_state=24)
    clf2.fit(X_train, y_train)
    
    # Check that importances are different
    assert not np.array_equal(clf1.feature_importances_, clf2.feature_importances_)


def test_feature_importances_with_single_dag():
    """Test feature importances with a single DAG."""
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Fit classifier with a single DAG
    clf = DecisionJungleClassifier(n_estimators=1, random_state=42)
    clf.fit(X_train, y_train)
    
    # Check if feature_importances_ is available
    assert hasattr(clf, 'feature_importances_')
    
    # Check dimensions
    assert clf.feature_importances_.shape == (X.shape[1],)
    
    # Check that importances sum to 1
    assert_allclose(np.sum(clf.feature_importances_), 1.0, rtol=1e-6)


def test_feature_importances_zero_sum():
    """Test handling of the edge case where no features are used (zero sum)."""
    # Create a simple dataset with identical samples
    X = np.ones((10, 5))
    y = np.zeros(10, dtype=int)
    
    # This should create DAGs with only leaf nodes (no splits)
    clf = DecisionJungleClassifier(n_estimators=1, random_state=42)
    clf.fit(X, y)
    
    # Check if feature_importances_ is available
    assert hasattr(clf, 'feature_importances_')
    
    # All importances should be zero
    assert_array_equal(clf.feature_importances_, np.zeros(5))


def test_feature_importances_not_fitted():
    """Test that feature_importances_ raises an error when not fitted."""
    clf = DecisionJungleClassifier()
    
    # Check that accessing feature_importances_ before fitting raises an error
    with pytest.raises(Exception):
        importances = clf.feature_importances_


def test_feature_importances_relative_values():
    """Test that feature importances have expected relative values on a synthetic dataset."""
    # Create a synthetic dataset where the first feature is highly predictive
    rng = np.random.RandomState(42)
    n_samples = 1000
    
    # First feature is strongly correlated with target
    X1 = rng.normal(0, 1, size=n_samples)
    
    # Other features are random noise
    X2 = rng.normal(0, 1, size=n_samples)
    X3 = rng.normal(0, 1, size=n_samples)
    X4 = rng.normal(0, 1, size=n_samples)
    
    # Create target based mostly on X1
    y = (X1 > 0).astype(int)
    
    # Combine features
    X = np.column_stack([X1, X2, X3, X4])
    
    # Fit the classifier
    clf = DecisionJungleClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    # First feature should have higher importance
    assert clf.feature_importances_[0] > clf.feature_importances_[1]
    assert clf.feature_importances_[0] > clf.feature_importances_[2]
    assert clf.feature_importances_[0] > clf.feature_importances_[3]