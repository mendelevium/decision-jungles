"""
Node classes for Decision Jungles.

This module implements the Node base class and its two primary subclasses: 
SplitNode and LeafNode that make up the nodes of a Decision DAG.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set, Any


class Node(ABC):
    """
    Abstract base class for nodes in a Decision DAG.
    
    Attributes:
        node_id (int): Unique identifier for the node.
        depth (int): Depth of the node in the DAG.
    """
    
    def __init__(self, node_id: int, depth: int):
        """
        Initialize a new Node.
        
        Args:
            node_id (int): Unique identifier for the node.
            depth (int): Depth of the node in the DAG.
        """
        self.node_id = node_id
        self.depth = depth
        self.parent_nodes: Set[int] = set()  # IDs of parent nodes
        
    def __getstate__(self) -> Dict[str, Any]:
        """
        Return the state of the node for pickling.
        
        Returns:
            dict: The state of the node to be serialized.
        """
        return self.__dict__.copy()
        
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Set the state of the node after unpickling.
        
        Args:
            state (dict): The state of the node as returned by __getstate__().
        """
        self.__dict__.update(state)
        
    def add_parent(self, parent_id: int) -> None:
        """
        Add a parent node to this node.
        
        Args:
            parent_id (int): The ID of the parent node to add.
        """
        self.parent_nodes.add(parent_id)
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make a prediction for the input samples.
        
        Args:
            X (np.ndarray): The input samples, shape (n_samples, n_features).
            
        Returns:
            For a SplitNode: Tuple of two arrays indicating samples that go left and right.
            For a LeafNode: Array of class probabilities.
        """
        pass
        
    @abstractmethod
    def get_memory_usage(self) -> int:
        """
        Calculate the memory usage of this node in bytes.
        
        Returns:
            int: Memory usage in bytes.
        """
        pass


class SplitNode(Node):
    """
    An internal node in a Decision DAG that splits the data.
    
    Attributes:
        node_id (int): Unique identifier for the node.
        depth (int): Depth of the node in the DAG.
        feature_idx (int): Index of the feature used for splitting.
        threshold (float): Threshold value for the split.
        left_child (int): ID of the left child node.
        right_child (int): ID of the right child node.
        is_categorical (bool): Whether this node splits on a categorical feature.
        categories_left (set): For categorical features, the set of categories to send left.
    """
    
    def __init__(self, node_id: int, depth: int, feature_idx: int, threshold: float, 
                 is_categorical: bool = False, categories_left: Optional[Set[Any]] = None):
        """
        Initialize a new SplitNode.
        
        Args:
            node_id (int): Unique identifier for the node.
            depth (int): Depth of the node in the DAG.
            feature_idx (int): Index of the feature used for splitting.
            threshold (float): Threshold value for the split (for numerical features).
            is_categorical (bool): Whether the feature is categorical.
            categories_left (Set, optional): For categorical features, the set of category values
                that should go to the left child. If None, defaults to an empty set.
        """
        super().__init__(node_id, depth)
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left_child: Optional[int] = None
        self.right_child: Optional[int] = None
        self.is_categorical: bool = is_categorical
        self.categories_left: Set[Any] = categories_left if categories_left is not None else set()
        
    def set_children(self, left_child: int, right_child: int) -> None:
        """
        Set the child nodes of this split node.
        
        Args:
            left_child (int): ID of the left child node.
            right_child (int): ID of the right child node.
        """
        self.left_child = left_child
        self.right_child = right_child
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine which child node each sample should go to.
        
        Args:
            X (np.ndarray): The input samples, shape (n_samples, n_features).
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Boolean masks indicating which samples
                go to the left and right children respectively.
        """
        if self.is_categorical:
            # For categorical features, check if the value is in the categories_left set
            go_left = np.zeros(X.shape[0], dtype=bool)
            for i in range(X.shape[0]):
                feature_value = X[i, self.feature_idx]
                # Handle NaN values - send them to the right
                if np.isnan(feature_value):
                    go_left[i] = False
                else:
                    go_left[i] = feature_value in self.categories_left
        else:
            # For numerical features, use the standard threshold comparison
            go_left = X[:, self.feature_idx] <= self.threshold
            
        go_right = ~go_left
        
        return go_left, go_right
    
    def get_memory_usage(self) -> int:
        """
        Calculate the memory usage of this node in bytes.
        
        Returns:
            int: Memory usage in bytes.
        """
        # Basic memory usage
        memory = 6 * 8  # Node ID, depth, feature_idx, threshold, left_child, right_child
        
        # Add memory for categorical feature handling
        if self.is_categorical:
            # Boolean flag + set overhead + elements in the set
            memory += 1 + 16 + len(self.categories_left) * 8
        
        # Add parent_nodes set overhead
        memory += 16
        
        return memory  # Rough estimate, depends on implementation details


class LeafNode(Node):
    """
    A leaf node in a Decision DAG that provides predictions.
    
    Attributes:
        node_id (int): Unique identifier for the node.
        depth (int): Depth of the node in the DAG.
        class_distribution (np.ndarray): Distribution of classes at this leaf.
    """
    
    def __init__(self, node_id: int, depth: int, n_classes: int):
        """
        Initialize a new LeafNode.
        
        Args:
            node_id (int): Unique identifier for the node.
            depth (int): Depth of the node in the DAG.
            n_classes (int): Number of classes in the distribution.
        """
        super().__init__(node_id, depth)
        self.class_distribution = np.zeros(n_classes)
        self.n_samples = 0
        
    def update_distribution(self, y: np.ndarray) -> None:
        """
        Update the class distribution with new samples.
        
        Args:
            y (np.ndarray): Class labels of the samples.
        """
        for label in y:
            self.class_distribution[label] += 1
        self.n_samples += len(y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X (np.ndarray): The input samples, shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Class probabilities for each sample,
                shape (n_samples, n_classes).
        """
        if self.n_samples > 0:
            # Normalize to get probabilities
            probs = self.class_distribution / self.n_samples
        else:
            # If no samples, equal probability for all classes
            probs = np.ones_like(self.class_distribution) / len(self.class_distribution)
            
        # Repeat the probability distribution for each sample
        return np.tile(probs, (X.shape[0], 1))
    
    def get_memory_usage(self) -> int:
        """
        Calculate the memory usage of this node in bytes.
        
        Returns:
            int: Memory usage in bytes.
        """
        # Node ID (int) + depth (int) + n_samples (int) +
        # class_distribution (np.ndarray of float64) + parent_nodes set overhead
        return 3 * 8 + 8 * len(self.class_distribution) + 16  # Rough estimate
