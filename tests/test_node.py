"""
Tests for the Node classes.
"""

import numpy as np
import pytest
from decision_jungles.node import SplitNode, LeafNode


def test_split_node_initialization():
    """Test initialization of a SplitNode."""
    node = SplitNode(node_id=1, depth=2, feature_idx=3, threshold=0.5)
    
    assert node.node_id == 1
    assert node.depth == 2
    assert node.feature_idx == 3
    assert node.threshold == 0.5
    assert node.left_child is None
    assert node.right_child is None
    assert len(node.parent_nodes) == 0


def test_split_node_set_children():
    """Test setting child nodes for a SplitNode."""
    node = SplitNode(node_id=1, depth=2, feature_idx=3, threshold=0.5)
    node.set_children(left_child=2, right_child=3)
    
    assert node.left_child == 2
    assert node.right_child == 3


def test_split_node_add_parent():
    """Test adding a parent to a SplitNode."""
    node = SplitNode(node_id=1, depth=2, feature_idx=3, threshold=0.5)
    node.add_parent(5)
    
    assert 5 in node.parent_nodes
    assert len(node.parent_nodes) == 1
    
    # Adding the same parent again should not duplicate
    node.add_parent(5)
    assert len(node.parent_nodes) == 1
    
    # Add another parent
    node.add_parent(6)
    assert 6 in node.parent_nodes
    assert len(node.parent_nodes) == 2


def test_split_node_predict():
    """Test predicting with a SplitNode."""
    node = SplitNode(node_id=1, depth=2, feature_idx=0, threshold=0.5)
    
    # Create sample data
    X = np.array([
        [0.3, 1.0],
        [0.7, 2.0],
        [0.5, 3.0]
    ])
    
    # Test prediction
    go_left, go_right = node.predict(X)
    
    assert np.array_equal(go_left, np.array([True, False, True]))
    assert np.array_equal(go_right, np.array([False, True, False]))


def test_split_node_memory_usage():
    """Test memory usage calculation for a SplitNode."""
    node = SplitNode(node_id=1, depth=2, feature_idx=3, threshold=0.5)
    memory = node.get_memory_usage()
    
    # Memory usage should be a positive integer
    assert memory > 0
    assert isinstance(memory, int)


def test_leaf_node_initialization():
    """Test initialization of a LeafNode."""
    node = LeafNode(node_id=1, depth=3, n_classes=4)
    
    assert node.node_id == 1
    assert node.depth == 3
    assert len(node.class_distribution) == 4
    assert np.all(node.class_distribution == 0)
    assert node.n_samples == 0
    assert len(node.parent_nodes) == 0


def test_leaf_node_update_distribution():
    """Test updating the class distribution of a LeafNode."""
    node = LeafNode(node_id=1, depth=3, n_classes=4)
    
    # Update with some sample labels
    node.update_distribution(np.array([0, 1, 2, 1, 0, 3]))
    
    assert np.array_equal(node.class_distribution, np.array([2, 2, 1, 1]))
    assert node.n_samples == 6
    
    # Update with more samples
    node.update_distribution(np.array([1, 1, 1]))
    
    assert np.array_equal(node.class_distribution, np.array([2, 5, 1, 1]))
    assert node.n_samples == 9


def test_leaf_node_predict():
    """Test predicting with a LeafNode."""
    node = LeafNode(node_id=1, depth=3, n_classes=3)
    
    # Update with some sample labels
    node.update_distribution(np.array([0, 1, 2, 1, 0, 1]))
    
    # Create sample data (features don't matter for leaf nodes)
    X = np.array([
        [0.3, 1.0],
        [0.7, 2.0],
        [0.5, 3.0]
    ])
    
    # Test prediction
    probs = node.predict(X)
    
    # Expected probabilities: [2/6, 3/6, 1/6]
    expected = np.array([
        [2/6, 3/6, 1/6],
        [2/6, 3/6, 1/6],
        [2/6, 3/6, 1/6]
    ])
    
    assert probs.shape == (3, 3)
    assert np.allclose(probs, expected)


def test_leaf_node_predict_empty():
    """Test predicting with an empty LeafNode."""
    node = LeafNode(node_id=1, depth=3, n_classes=3)
    
    # No samples yet
    assert node.n_samples == 0
    
    # Create sample data
    X = np.array([
        [0.3, 1.0],
        [0.7, 2.0]
    ])
    
    # Test prediction
    probs = node.predict(X)
    
    # Expected: uniform distribution
    expected = np.array([
        [1/3, 1/3, 1/3],
        [1/3, 1/3, 1/3]
    ])
    
    assert probs.shape == (2, 3)
    assert np.allclose(probs, expected)


def test_leaf_node_memory_usage():
    """Test memory usage calculation for a LeafNode."""
    node = LeafNode(node_id=1, depth=3, n_classes=10)
    memory = node.get_memory_usage()
    
    # Memory usage should be a positive integer
    assert memory > 0
    assert isinstance(memory, int)
    
    # Memory should be greater for more classes
    node2 = LeafNode(node_id=2, depth=3, n_classes=20)
    memory2 = node2.get_memory_usage()
    
    assert memory2 > memory
