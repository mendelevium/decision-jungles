"""
Tests for the DAG class.
"""

import numpy as np
import pytest
from decision_jungles.dag import DAG
from decision_jungles.node import SplitNode, LeafNode


def test_dag_initialization():
    """Test initialization of a DAG."""
    dag = DAG(n_classes=3, max_depth=10, max_width=128, random_state=42)
    
    assert dag.n_classes == 3
    assert dag.max_depth == 10
    assert dag.max_width == 128
    assert dag.random_state == 42
    assert len(dag.nodes) == 0
    assert dag.root_node_id is None


def test_dag_fit_trivial():
    """Test fitting a DAG with a trivial dataset."""
    # Create a tiny dataset with a single class
    X = np.array([[0.5, 0.5]])
    y = np.array([0])
    
    dag = DAG(n_classes=1, max_depth=3, max_width=128, random_state=42)
    dag.fit(X, y)
    
    # Should create a single leaf node
    assert len(dag.nodes) == 1
    assert dag.root_node_id is not None
    assert isinstance(dag.nodes[dag.root_node_id], LeafNode)
    
    # The root node should have the correct class distribution
    root_node = dag.nodes[dag.root_node_id]
    assert np.array_equal(root_node.class_distribution, np.array([1]))
    assert root_node.n_samples == 1


def test_dag_fit_simple():
    """Test fitting a DAG with a simple dataset."""
    # Create a small dataset with two well-separated classes
    X = np.array([
        [0.3, 0.3],
        [0.2, 0.2],
        [0.8, 0.8],
        [0.9, 0.9]
    ])
    y = np.array([0, 0, 1, 1])
    
    dag = DAG(n_classes=2, max_depth=3, max_width=128, random_state=42)
    dag.fit(X, y)
    
    # Should create a root split node and at least two child nodes
    assert len(dag.nodes) >= 3
    assert dag.root_node_id is not None
    assert isinstance(dag.nodes[dag.root_node_id], SplitNode)
    
    # Predict should return correct classes
    predictions = dag.predict(X)
    assert np.array_equal(predictions, y)


def test_dag_merging_schedule():
    """Test different merging schedules."""
    # Create a dataset
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)
    
    # Test constant schedule
    dag1 = DAG(n_classes=3, max_depth=5, max_width=8, merging_schedule="constant", random_state=42)
    dag1.fit(X, y)
    
    # Test exponential schedule
    dag2 = DAG(n_classes=3, max_depth=5, max_width=32, merging_schedule="exponential", random_state=42)
    dag2.fit(X, y)
    
    # Test kinect schedule
    dag3 = DAG(n_classes=3, max_depth=5, max_width=32, merging_schedule="kinect", random_state=42)
    dag3.fit(X, y)
    
    # Each merging schedule should produce a valid DAG
    assert len(dag1.nodes) > 0
    assert len(dag2.nodes) > 0
    assert len(dag3.nodes) > 0
    
    # Exponential schedule should produce more nodes than constant
    assert len(dag2.nodes) > len(dag1.nodes)


def test_dag_predict_proba():
    """Test probability predictions from a DAG."""
    # Create a dataset with two separated classes
    X = np.array([
        [0.3, 0.3],
        [0.2, 0.2],
        [0.8, 0.8],
        [0.9, 0.9]
    ])
    y = np.array([0, 0, 1, 1])
    
    dag = DAG(n_classes=2, max_depth=3, max_width=128, random_state=42)
    dag.fit(X, y)
    
    # Get probability predictions
    proba = dag.predict_proba(X)
    
    # Check shape and properties
    assert proba.shape == (4, 2)  # 4 samples, 2 classes
    assert np.all(proba >= 0)  # Probabilities should be non-negative
    assert np.all(proba <= 1)  # Probabilities should be at most 1
    assert np.allclose(np.sum(proba, axis=1), 1.0)  # Probabilities should sum to 1
    
    # Check predictions based on probabilities
    pred_from_proba = np.argmax(proba, axis=1)
    pred_direct = dag.predict(X)
    assert np.array_equal(pred_from_proba, pred_direct)
    assert np.array_equal(pred_direct, y)  # Should get original labels


def test_dag_memory_usage():
    """Test memory usage calculation."""
    # Create a small dataset
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 3, 50)
    
    dag = DAG(n_classes=3, max_depth=3, max_width=16, random_state=42)
    dag.fit(X, y)
    
    # Calculate memory usage
    memory = dag.get_memory_usage()
    
    # Memory usage should be positive and increase with more nodes
    assert memory > 0
    
    # Create a larger DAG
    dag2 = DAG(n_classes=3, max_depth=5, max_width=32, random_state=42)
    dag2.fit(X, y)
    
    memory2 = dag2.get_memory_usage()
    
    # The larger DAG should use at least as much memory
    assert memory2 >= memory


def test_dag_max_depth():
    """Test respecting the max_depth parameter."""
    # Create a dataset
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)
    
    # Set different max depths
    dag1 = DAG(n_classes=3, max_depth=2, max_width=128, random_state=42)
    dag1.fit(X, y)
    
    dag2 = DAG(n_classes=3, max_depth=4, max_width=128, random_state=42)
    dag2.fit(X, y)
    
    # Check max depths
    assert dag1.get_max_depth() <= 2
    assert dag2.get_max_depth() <= 4
    
    # The deeper DAG should have at least as many nodes
    assert dag2.get_node_count() >= dag1.get_node_count()


def test_dag_min_samples_split():
    """Test respecting the min_samples_split parameter."""
    # Create a dataset
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)
    
    # Set different min_samples_split
    dag1 = DAG(n_classes=3, max_depth=5, max_width=128, min_samples_split=2, random_state=42)
    dag1.fit(X, y)
    
    dag2 = DAG(n_classes=3, max_depth=5, max_width=128, min_samples_split=20, random_state=42)
    dag2.fit(X, y)
    
    # The DAG with higher min_samples_split should have at most as many nodes
    assert dag2.get_node_count() <= dag1.get_node_count()


def test_dag_node_count():
    """Test node counting."""
    # Create a dataset
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)
    
    dag = DAG(n_classes=2, max_depth=3, max_width=16, random_state=42)
    dag.fit(X, y)
    
    # Node count should match the length of nodes dictionary
    assert dag.get_node_count() == len(dag.nodes)
