"""
Tests for objective functions.
"""

import numpy as np
import pytest
from decision_jungles.training.objective import entropy, weighted_entropy_sum, information_gain, optimize_split


def test_entropy_empty():
    """Test entropy calculation with empty array."""
    result = entropy(np.array([]))
    assert result == 0.0


def test_entropy_single_class():
    """Test entropy calculation with single class."""
    result = entropy(np.array([1, 1, 1, 1, 1]))
    assert result == 0.0


def test_entropy_uniform_distribution():
    """Test entropy calculation with uniform distribution."""
    result = entropy(np.array([0, 1, 2, 3]))
    # For 4 classes with uniform distribution, H = -log2(1/4)
    expected = -np.log2(0.25) * 0.25 * 4
    assert np.isclose(result, expected, rtol=1e-6)


def test_entropy_non_uniform():
    """Test entropy calculation with non-uniform distribution."""
    result = entropy(np.array([0, 0, 1, 2, 2, 2]))
    
    # Manual calculation
    p0 = 2/6
    p1 = 1/6
    p2 = 3/6
    expected = -(p0 * np.log2(p0) + p1 * np.log2(p1) + p2 * np.log2(p2))
    
    assert np.isclose(result, expected, rtol=1e-6)


def test_entropy_with_n_classes():
    """Test entropy calculation with explicit n_classes."""
    result = entropy(np.array([0, 0, 1]), n_classes=3)
    
    # Manual calculation
    p0 = 2/3
    p1 = 1/3
    p2 = 0
    expected = -(p0 * np.log2(p0) + p1 * np.log2(p1))  # p2 term is 0
    
    assert np.isclose(result, expected, rtol=1e-6)


def test_weighted_entropy_sum_empty():
    """Test weighted entropy sum with empty list."""
    result = weighted_entropy_sum([])
    assert result == 0.0


def test_weighted_entropy_sum_empty_sets():
    """Test weighted entropy sum with empty sets."""
    result = weighted_entropy_sum([np.array([]), np.array([])])
    assert result == 0.0


def test_weighted_entropy_sum_single_set():
    """Test weighted entropy sum with a single set."""
    result = weighted_entropy_sum([np.array([0, 1, 2])])
    expected = entropy(np.array([0, 1, 2]))
    assert np.isclose(result, expected, rtol=1e-6)


def test_weighted_entropy_sum_multiple_sets():
    """Test weighted entropy sum with multiple sets."""
    set1 = np.array([0, 0, 1])
    set2 = np.array([1, 2, 2])
    
    result = weighted_entropy_sum([set1, set2])
    
    # Manual calculation
    entropy1 = entropy(set1)
    entropy2 = entropy(set2)
    weight1 = len(set1) / (len(set1) + len(set2))
    weight2 = len(set2) / (len(set1) + len(set2))
    expected = weight1 * entropy1 + weight2 * entropy2
    
    assert np.isclose(result, expected, rtol=1e-6)


def test_information_gain_empty():
    """Test information gain with empty parent."""
    result = information_gain(np.array([]), [np.array([]), np.array([])])
    assert result == 0.0


def test_information_gain_no_split():
    """Test information gain with no split (all samples in one child)."""
    parent = np.array([0, 0, 1, 2])
    children = [parent, np.array([])]
    
    result = information_gain(parent, children)
    
    # No split means no information gain
    assert np.isclose(result, 0.0, rtol=1e-6)


def test_information_gain_perfect_split():
    """Test information gain with perfect split."""
    parent = np.array([0, 0, 1, 1])
    child1 = np.array([0, 0])
    child2 = np.array([1, 1])
    
    result = information_gain(parent, [child1, child2])
    
    # Parent entropy - weighted children entropy
    parent_entropy = entropy(parent)
    # Each child has only one class, so entropy = 0
    expected = parent_entropy
    
    assert np.isclose(result, expected, rtol=1e-6)


def test_information_gain_partial_split():
    """Test information gain with partial split."""
    parent = np.array([0, 0, 0, 1, 1, 2])
    child1 = np.array([0, 0, 1])
    child2 = np.array([0, 1, 2])
    
    result = information_gain(parent, [child1, child2])
    
    # Parent entropy - weighted children entropy
    parent_entropy = entropy(parent)
    child1_entropy = entropy(child1)
    child2_entropy = entropy(child2)
    weighted_children_entropy = (len(child1) / len(parent)) * child1_entropy + (len(child2) / len(parent)) * child2_entropy
    expected = parent_entropy - weighted_children_entropy
    
    assert np.isclose(result, expected, rtol=1e-6)


def test_optimize_split_not_enough_samples():
    """Test split optimization with not enough samples."""
    X = np.array([[0.5]])
    y = np.array([0])
    
    result = optimize_split(X, y)
    
    # Not enough samples to split
    assert result[0] == -1
    assert result[1] == 0.0
    assert result[2] == 0.0


def test_optimize_split_single_class():
    """Test split optimization with single class."""
    X = np.array([
        [0.3, 1.0],
        [0.7, 2.0],
        [0.5, 3.0]
    ])
    y = np.array([0, 0, 0])
    
    feature_idx, threshold, info_gain, is_cat, cats = optimize_split(X, y)
    
    # Single class, so no information gain
    assert info_gain == 0.0


def test_optimize_split_perfect_separation():
    """Test split optimization with perfect class separation."""
    X = np.array([
        [0.3, 1.0],
        [0.2, 2.0],
        [0.7, 3.0],
        [0.8, 4.0]
    ])
    y = np.array([0, 0, 1, 1])
    
    feature_idx, threshold, info_gain, is_cat, cats = optimize_split(X, y)
    
    # Should find a threshold that perfectly separates classes
    assert feature_idx == 0  # First feature separates classes
    assert 0.3 < threshold < 0.7  # Threshold should be between 0.3 and 0.7
    
    # Verify split quality
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    
    y_left = y[left_mask]
    y_right = y[right_mask]
    
    # Should be perfect separation
    assert np.all(y_left == 0)
    assert np.all(y_right == 1)
    
    # Information gain should be equal to parent entropy
    parent_entropy = entropy(y)
    assert np.isclose(info_gain, parent_entropy, rtol=1e-6)


def test_optimize_split_with_feature_subset():
    """Test split optimization with a subset of features."""
    X = np.array([
        [0.3, 9.0],
        [0.2, 8.0],
        [0.7, 2.0],
        [0.8, 1.0]
    ])
    y = np.array([0, 0, 1, 1])
    
    # Use only the second feature
    feature_idx, threshold, info_gain, is_cat, cats = optimize_split(X, y, feature_indices=[1])
    
    # Should use the second feature
    assert feature_idx == 1
    assert 2.0 < threshold < 8.0  # Threshold should be between 2.0 and 8.0
    
    # Verify split quality
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    
    y_left = y[left_mask]
    y_right = y[right_mask]
    
    # Should be perfect separation
    assert np.all(y_left == 1)
    assert np.all(y_right == 0)
    
    # Information gain should be equal to parent entropy
    parent_entropy = entropy(y)
    assert np.isclose(info_gain, parent_entropy, rtol=1e-6)
