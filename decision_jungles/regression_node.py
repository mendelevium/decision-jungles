"""
Node classes for Regression Decision Jungles.

This module implements regression-specific node classes for Decision Jungles,
particularly the RegressionLeafNode that stores continuous target values.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set, Any

from .node import Node, SplitNode


class RegressionLeafNode(Node):
    """
    A leaf node in a regression Decision DAG that provides predictions.
    
    Unlike the classification LeafNode which stores class distributions,
    this node stores the mean target value and provides continuous predictions.
    
    Attributes:
        node_id (int): Unique identifier for the node.
        depth (int): Depth of the node in the DAG.
        target_value (float): Mean target value at this leaf.
        n_samples (int): Number of samples at this leaf.
    """
    
    def __init__(self, node_id: int, depth: int):
        """
        Initialize a new RegressionLeafNode.
        
        Args:
            node_id (int): Unique identifier for the node.
            depth (int): Depth of the node in the DAG.
        """
        super().__init__(node_id, depth)
        self.target_sum = 0.0
        self.n_samples = 0
        self.target_value = 0.0
        
    def update_value(self, y: np.ndarray) -> None:
        """
        Update the target value with new samples.
        
        Args:
            y (np.ndarray): Target values of the samples.
        """
        if len(y) > 0:
            self.target_sum += np.sum(y)
            self.n_samples += len(y)
            self.target_value = self.target_sum / self.n_samples
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples.
        
        Args:
            X (np.ndarray): The input samples, shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted target values for each sample, shape (n_samples,).
        """
        # Return the same value for all samples
        return np.full(X.shape[0], self.target_value)
    
    def get_memory_usage(self) -> int:
        """
        Calculate the memory usage of this node in bytes.
        
        Returns:
            int: Memory usage in bytes.
        """
        # Node ID (int) + depth (int) + n_samples (int) + target_sum (float) + 
        # target_value (float) + parent_nodes set overhead
        return 5 * 8 + 16  # Rough estimate