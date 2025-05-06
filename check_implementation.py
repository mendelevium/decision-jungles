"""
Script to check if the Decision Jungle implementation is working correctly.

This script performs a simple test to verify that all components are working
together as expected.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import our modules
from decision_jungles.jungle import DecisionJungleClassifier
from decision_jungles.dag import DAG
from decision_jungles.node import SplitNode, LeafNode


def check_node_classes():
    """Check if Node classes are working correctly."""
    print("Testing Node classes...")
    
    # Create a split node
    split_node = SplitNode(node_id=1, depth=0, feature_idx=0, threshold=0.5)
    split_node.set_children(left_child=2, right_child=3)
    
    # Create test data
    X = np.array([[0.3], [0.7]])
    go_left, go_right = split_node.predict(X)
    
    assert go_left[0] and not go_left[1], "Split prediction failed"
    assert not go_right[0] and go_right[1], "Split prediction failed"
    
    # Create a leaf node
    leaf_node = LeafNode(node_id=2, depth=1, n_classes=3)
    leaf_node.update_distribution(np.array([0, 1, 0, 2, 0]))
    
    # Test prediction
    probs = leaf_node.predict(X)
    
    assert probs.shape == (2, 3), "Leaf prediction shape is incorrect"
    assert np.isclose(probs[0, 0], 3/5), "Leaf prediction probabilities are incorrect"
    assert np.isclose(probs[0, 1], 1/5), "Leaf prediction probabilities are incorrect"
    assert np.isclose(probs[0, 2], 1/5), "Leaf prediction probabilities are incorrect"
    
    print("Node classes test passed!")


def check_dag_class():
    """Check if DAG class is working correctly."""
    print("Testing DAG class...")
    
    # Create a simple dataset
    X = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.8, 0.8],
        [0.9, 0.9]
    ])
    y = np.array([0, 0, 1, 1])
    
    # Create and fit a DAG
    dag = DAG(n_classes=2, max_depth=3, max_width=4, random_state=42)
    dag.fit(X, y)
    
    # Check if nodes were created
    assert len(dag.nodes) > 0, "No nodes were created"
    assert dag.root_node_id is not None, "Root node was not set"
    
    # Check predictions
    predictions = dag.predict(X)
    assert np.array_equal(predictions, y), "DAG predictions are incorrect"
    
    print("DAG class test passed!")


def check_jungle_class():
    """Check if DecisionJungleClassifier is working correctly."""
    print("Testing DecisionJungleClassifier...")
    
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and fit a jungle
    jungle = DecisionJungleClassifier(
        n_estimators=3,
        max_width=16,
        max_depth=5,
        random_state=42
    )
    jungle.fit(X_train, y_train)
    
    # Check if DAGs were created
    assert hasattr(jungle, 'dags_'), "DAGs were not created"
    assert len(jungle.dags_) == 3, "Wrong number of DAGs"
    
    # Check predictions
    y_pred = jungle.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Jungle achieved {accuracy:.4f} accuracy on Iris test set")
    assert accuracy > 0.8, "Jungle accuracy is too low"
    
    # Check probability predictions
    proba = jungle.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3), "Wrong probability shape"
    assert np.all(proba >= 0) and np.all(proba <= 1), "Invalid probabilities"
    assert np.allclose(np.sum(proba, axis=1), 1.0), "Probabilities don't sum to 1"
    
    # Check utility methods
    n_nodes = jungle.get_node_count()
    assert n_nodes > 0, "Node count failed"
    
    memory = jungle.get_memory_usage()
    assert memory > 0, "Memory usage calculation failed"
    
    max_depth = jungle.get_max_depth()
    assert max_depth > 0 and max_depth <= 5, "Max depth calculation failed"
    
    print("DecisionJungleClassifier test passed!")


def check_lsearch_optimization():
    """Check if LSearch optimization is working correctly."""
    print("Testing LSearch optimization...")
    
    # Create a dataset with a clearer separation between classes
    X = np.array([
        [0.1, 0.1],  # Class 0
        [0.2, 0.2],  # Class 0
        [0.3, 0.3],  # Class 0
        [0.4, 0.4],  # Class 0
        [0.8, 0.8],  # Class 1
        [0.9, 0.9],  # Class 1
        [0.7, 0.7],  # Class 1
        [0.6, 0.6]   # Class 1
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Regular tree should create many nodes
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    tree_nodes = tree.tree_.node_count
    
    # DAG should merge similar nodes - use different configuration for better merging
    # Reduce max_width and increase depth to force more node merging
    dag = DAG(n_classes=2, max_depth=3, max_width=2, random_state=42, merging_schedule="constant")
    dag.fit(X, y)
    dag_nodes = dag.get_node_count()
    
    print(f"Tree nodes: {tree_nodes}, DAG nodes: {dag_nodes}")
    assert dag_nodes <= tree_nodes, "DAG should have the same or fewer nodes than tree"
    
    # Check predictions
    predictions = dag.predict(X)
    assert np.array_equal(predictions, y), "DAG predictions after merging are incorrect"
    
    print("LSearch optimization test passed!")


def run_all_checks():
    """Run all implementation checks."""
    print("=== Checking Decision Jungle Implementation ===\n")
    
    try:
        check_node_classes()
        print()
        
        check_dag_class()
        print()
        
        check_jungle_class()
        print()
        
        check_lsearch_optimization()
        print()
        
        print("=== All checks passed! ===")
        print("The Decision Jungle implementation appears to be working correctly.")
        
    except AssertionError as e:
        print(f"Error: {e}")
        print("Some checks failed. Please check the implementation.")


if __name__ == "__main__":
    run_all_checks()
