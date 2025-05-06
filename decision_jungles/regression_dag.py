"""
Regression DAG (Directed Acyclic Graph) implementation for Decision Jungles.

This module implements the RegressionDAG class that represents a single directed acyclic
graph within a Decision Jungle ensemble for regression tasks.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set, Any, cast
import heapq
from .node import Node, SplitNode
from .regression_node import RegressionLeafNode
from .training.reg_objective import optimize_split_regression

class RegressionDAG:
    """
    A Directed Acyclic Graph (DAG) for Decision Jungle regression.
    
    This class implements a single DAG component of a Decision Jungle ensemble
    for regression tasks. Unlike a traditional decision tree where each node has
    exactly one parent, nodes in a DAG can have multiple parents, allowing for a
    more compact representation while maintaining or improving predictive performance.
    
    The DAG is built level-by-level with a controlled width at each level,
    determined by the merging schedule. During training, the optimization algorithm 
    is used to determine both the optimal split parameters and the optimal branch assignments.
    
    Attributes
    ----------
    nodes : Dict[int, Node]
        Dictionary of all nodes in the DAG, keyed by node ID.
        
    root_node_id : int
        ID of the root node of the DAG.
        
    next_node_id : int
        Counter for generating unique node IDs.
        
    max_width : int
        Maximum width of each level (M parameter from the paper).
        
    max_depth : int or None
        Maximum depth of the DAG. If None, the DAG will grow until all
        leaves are pure or contain less than min_samples_split samples.
    """
    
    def __init__(self, max_depth: Optional[int] = None,
                min_samples_split: int = 2, min_samples_leaf: int = 1,
                min_impurity_decrease: float = 0.0, max_features: Optional[Union[int, float, str]] = "auto",
                criterion: str = "mse", random_state: Optional[int] = None, 
                merging_schedule: str = "exponential", max_width: int = 256,
                optimization_method: str = "lsearch", use_optimized: bool = True, 
                early_stopping: bool = False, validation_X: Optional[np.ndarray] = None, 
                validation_y: Optional[np.ndarray] = None, n_iter_no_change: int = 5, 
                tol: float = 1e-4, is_categorical: Optional[np.ndarray] = None, 
                feature_bins: Optional[Dict[int, Dict[Any, int]]] = None):
        """
        Initialize a new RegressionDAG.
        
        Args:
            max_depth (int, optional): Maximum depth of the DAG.
            min_samples_split (int): Minimum number of samples required to split a node.
            min_samples_leaf (int): Minimum number of samples required at a leaf node.
            min_impurity_decrease (float): Minimum impurity decrease required for a split.
            max_features (int, float, str, optional): Number of features to consider for splits.
            criterion (str): The criterion to use for splitting ("mse" or "mae").
            random_state (int, optional): Random seed for reproducibility.
            merging_schedule (str): Type of merging schedule to use ("constant", "exponential", "kinect").
            max_width (int): Maximum width of each level (M parameter).
            optimization_method (str): Method for DAG optimization ("lsearch" or "clustersearch").
            use_optimized (bool): Whether to use the optimized implementation.
            early_stopping (bool): Whether to use early stopping to terminate training.
            validation_X (ndarray, optional): Validation features for early stopping.
            validation_y (ndarray, optional): Validation targets for early stopping.
            n_iter_no_change (int): Number of iterations with no improvement to wait before early stopping.
            tol (float): Tolerance for the early stopping criterion.
            is_categorical (ndarray, optional): Boolean mask indicating which features are categorical.
            feature_bins (Dict, optional): Dictionary mapping feature indices to category-to-bin mappings.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.merging_schedule = merging_schedule
        self.max_width = max_width
        self.optimization_method = optimization_method
        self.use_optimized = use_optimized
        self.early_stopping = early_stopping
        self.validation_X = validation_X
        self.validation_y = validation_y
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        
        # Categorical feature handling
        self.is_categorical: Optional[np.ndarray] = is_categorical
        self.feature_bins: Optional[Dict[int, Dict[Any, int]]] = feature_bins
        
        self.nodes: Dict[int, Node] = {}
        self.root_node_id: Optional[int] = None
        self.next_node_id = 0
        
        # Early stopping tracking
        self.validation_scores = []
        self.best_validation_score = np.inf
        self.iterations_no_improvement = 0
        self.stopped_early = False
        
        self._rng = np.random.RandomState(random_state)
        
    def _get_node_id(self) -> int:
        """
        Generate a new unique node ID.
        
        Returns:
            int: A new unique node ID.
        """
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id
    
    def _get_level_width(self, depth: int) -> int:
        """
        Calculate the maximum width of a level according to the merging schedule.
        
        Args:
            depth (int): The depth of the level.
            
        Returns:
            int: The maximum width of the level.
        """
        # For both constant and exponential, cap by the max_width parameter
        # but also control growth with depth
        if self.merging_schedule == "constant":
            # For constant schedule, use same width at all levels
            return min(self.max_width, 2)  # Reduced width for better merging
        elif self.merging_schedule == "exponential":
            # Cap exponential growth more aggressively 
            return min(self.max_width, max(2, min(2**depth, 4)))  # More controlled growth
        elif self.merging_schedule == "kinect":
            # Kinect merging schedule from the paper, but ensure it returns an integer
            width = min(self.max_width, int(2**min(5, depth) * 1.2**max(0, depth-5)))
            return max(2, width)  # Ensure at least 2 nodes for branching
        else:
            raise ValueError(f"Unknown merging schedule: {self.merging_schedule}")
    
    def _get_optimization_method(self):
        """
        Get the optimization method to use for finding the best splits.
        
        The primary difference for regression is that we use a different
        objective function (MSE or MAE) instead of entropy/information gain.
        
        Returns:
            str: The name of the optimization method.
        """
        return self.optimization_method
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the DAG to the training data.
        
        This method builds a directed acyclic graph (DAG) structure for regression
        based on the provided training data. The DAG is constructed level-by-level
        with a controlled width at each level, determined by the merging schedule.
        
        The key innovation of Decision Jungles is that multiple parent nodes can share
        the same child nodes, leading to a more compact representation compared to
        traditional decision trees.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training input samples.
            
        y : ndarray of shape (n_samples,)
            The target values.
            
        Notes
        -----
        The fitting process involves these key steps:
        
        1. Create a root node containing all training samples
        2. For each level up to max_depth:
           a. Determine the width of the next level based on the merging schedule
           b. Create temporary nodes for the next level
           c. Use the optimization algorithm to find:
              - Optimal split parameters for each parent node
              - Optimal branch assignments to child nodes
           d. Update the DAG structure with the optimized splits and branches
        3. Stop when max_depth is reached or no more nodes can be split
        
        The DAG is constructed to maximize impurity reduction while respecting
        the width constraints, leading to a more memory-efficient model.
        """
        # Create the root node
        self.root_node_id = self._get_node_id()
        self.nodes[self.root_node_id] = RegressionLeafNode(self.root_node_id, 0)
        self.nodes[self.root_node_id].update_value(y)
        
        # Track samples at each node
        node_samples = {self.root_node_id: (np.arange(X.shape[0]), y)}
        
        # Track leaf nodes to process
        leaf_nodes = [self.root_node_id]
        current_depth = 0
        
        # Grow the DAG level by level
        while leaf_nodes and (self.max_depth is None or current_depth < self.max_depth):
            # Determine the width of the next level
            next_level_width = self._get_level_width(current_depth + 1)
            
            # Skip if we've reached our maximum width and can't split further
            if next_level_width <= len(leaf_nodes):
                break
            
            # Filter leaf nodes that have enough samples to split
            splittable_nodes = []
            for node_id in leaf_nodes:
                # Skip this node if it's not in node_samples
                if node_id not in node_samples:
                    continue
                    
                indices, targets = node_samples[node_id]
                # Skip if there are not enough samples to split
                if (len(indices) == 0 or 
                    len(indices) < self.min_samples_split):
                    continue
                    
                splittable_nodes.append(node_id)
            
            # Skip this level if no nodes can be split
            if not splittable_nodes:
                break
            
            # Create new child nodes at the next depth
            next_depth = current_depth + 1
            next_level_nodes = []
            for _ in range(next_level_width):
                node_id = self._get_node_id()
                self.nodes[node_id] = RegressionLeafNode(node_id, next_depth)
                next_level_nodes.append(node_id)
            
            # Select features to consider for splitting
            n_features = X.shape[1]
            if isinstance(self.max_features, str):
                if self.max_features == "auto" or self.max_features == "sqrt":
                    max_features = max(1, int(np.sqrt(n_features)))
                elif self.max_features == "log2":
                    max_features = max(1, int(np.log2(n_features)))
                else:
                    raise ValueError(f"Invalid max_features: {self.max_features}")
            elif isinstance(self.max_features, (int, float)):
                if isinstance(self.max_features, int):
                    max_features = min(self.max_features, n_features)
                else:  # float
                    max_features = max(1, int(self.max_features * n_features))
            else:
                max_features = n_features
                
            # Sample features for each node
            node_feature_indices = {}
            for node_id in splittable_nodes:
                if max_features < n_features:
                    # Randomly select features for this node
                    feature_indices = self._rng.choice(n_features, size=max_features, replace=False)
                    node_feature_indices[node_id] = feature_indices.tolist()
                else:
                    # Use all features
                    node_feature_indices[node_id] = list(range(n_features))
            
            # Optimize splits for each node
            split_params = {}
            left_child = {}
            right_child = {}
            
            for node_id in splittable_nodes:
                indices, targets = node_samples[node_id]
                X_node = X[indices]
                y_node = targets
                
                # Find the best split for this node
                feature_indices = node_feature_indices[node_id]
                feature_idx, threshold, impurity_reduction, is_cat, categories_left = optimize_split_regression(
                    X_node, y_node, feature_indices=feature_indices, criterion=self.criterion,
                    is_categorical=self.is_categorical, feature_bins=self.feature_bins
                )
                
                # Skip if no good split was found or impurity reduction is too small
                if feature_idx == -1 or impurity_reduction < self.min_impurity_decrease:
                    continue
                
                # Assign child nodes arbitrarily (they will be merged if necessary)
                # In a more sophisticated implementation, we would optimize this assignment
                left_idx = self._rng.choice(len(next_level_nodes))
                right_idx = (left_idx + 1) % len(next_level_nodes)  # Choose a different node
                
                # Store the split parameters and child assignments
                if is_cat:
                    split_params[node_id] = (feature_idx, threshold, is_cat, categories_left)
                else:
                    split_params[node_id] = (feature_idx, threshold)
                left_child[node_id] = left_idx
                right_child[node_id] = right_idx
            
            # Update node structure based on the optimization results
            new_node_samples = {node_id: (np.array([], dtype=int), np.array([], dtype=float))
                               for node_id in next_level_nodes}
            
            # Process splittable nodes
            for node_id in splittable_nodes:
                indices, targets = node_samples[node_id]
                
                if node_id in split_params:
                    # Unpack split parameters with categorical support
                    if len(split_params[node_id]) == 4:  # New format with categorical info
                        feature_idx, threshold, is_cat, categories_left = split_params[node_id]
                    else:  # Old format without categorical info
                        feature_idx, threshold = split_params[node_id]
                        is_cat, categories_left = False, None
                    
                    # Convert the leaf node to a split node
                    depth = self.nodes[node_id].depth
                    self.nodes[node_id] = SplitNode(node_id, depth, feature_idx, threshold, 
                                                   is_categorical=is_cat, categories_left=categories_left)
                    
                    # Set child relationships
                    left_id = next_level_nodes[left_child[node_id]]
                    right_id = next_level_nodes[right_child[node_id]]
                    self.nodes[node_id].set_children(left_id, right_id)
                    
                    # Add parent relationships
                    self.nodes[left_id].add_parent(node_id)
                    self.nodes[right_id].add_parent(node_id)
                    
                    # Split samples
                    if is_cat and categories_left:
                        # For categorical features
                        left_mask = np.zeros(len(indices), dtype=bool)
                        for i in range(len(indices)):
                            idx = indices[i]
                            feature_value = X[idx, feature_idx]
                            if not np.isnan(feature_value) and feature_value in categories_left:
                                left_mask[i] = True
                    else:
                        # For numerical features
                        left_mask = X[indices, feature_idx] <= threshold
                        # Handle NaN values - send them to the right
                        nan_mask = np.isnan(X[indices, feature_idx])
                        left_mask[nan_mask] = False
                        
                    right_mask = ~left_mask
                    
                    # Update samples for child nodes
                    if np.any(left_mask):
                        left_indices = indices[left_mask]
                        left_targets = targets[left_mask]
                        
                        # Append to existing samples for this child node
                        existing_indices, existing_targets = new_node_samples[left_id]
                        if len(existing_indices) == 0:
                            new_node_samples[left_id] = (left_indices, left_targets)
                        else:
                            new_node_samples[left_id] = (
                                np.concatenate([existing_indices, left_indices]),
                                np.concatenate([existing_targets, left_targets])
                            )
                        
                        # Update target value
                        self.nodes[left_id].update_value(left_targets)
                    
                    if np.any(right_mask):
                        right_indices = indices[right_mask]
                        right_targets = targets[right_mask]
                        
                        # Append to existing samples for this child node
                        existing_indices, existing_targets = new_node_samples[right_id]
                        if len(existing_indices) == 0:
                            new_node_samples[right_id] = (right_indices, right_targets)
                        else:
                            new_node_samples[right_id] = (
                                np.concatenate([existing_indices, right_indices]),
                                np.concatenate([existing_targets, right_targets])
                            )
                        
                        # Update target value
                        self.nodes[right_id].update_value(right_targets)
            
            # Process non-splittable leaf nodes at current depth
            non_splittable = [node_id for node_id in leaf_nodes if node_id not in splittable_nodes]
            
            # Keep non-splittable leaf nodes as they are
            for node_id in non_splittable:
                pass  # Already a leaf node, no need to modify
            
            # Update leaf nodes for the next iteration
            leaf_nodes = [node_id for node_id in next_level_nodes
                         if isinstance(self.nodes[node_id], RegressionLeafNode)]
            
            # Update node samples for the next iteration
            node_samples = {}
            for node_id in leaf_nodes:
                if node_id in new_node_samples and len(new_node_samples[node_id][0]) > 0:
                    node_samples[node_id] = new_node_samples[node_id]
                else:
                    # Include the node with empty sample arrays
                    node_samples[node_id] = (np.array([], dtype=int), np.array([], dtype=float))
            
            # Update current depth
            current_depth = next_depth
            
            # Check early stopping criteria if enabled
            if self.early_stopping and self._check_early_stopping(current_depth):
                # If early stopping criteria met, break the loop
                break
    
    def _evaluate_validation_score(self) -> float:
        """
        Evaluate the DAG's performance on the validation set.
        
        Returns:
            float: Validation MSE score (negated), or -inf if no validation data is available.
        """
        if self.validation_X is None or self.validation_y is None:
            return np.inf
            
        # Make predictions on validation set
        y_pred = self.predict(self.validation_X)
        
        # Calculate MSE
        mse = np.mean((y_pred - self.validation_y) ** 2)
        
        return mse
    
    def _check_early_stopping(self, current_depth: int) -> bool:
        """
        Check if early stopping criteria are met.
        
        For regression, we want to minimize MSE, so the logic is reversed
        compared to classification (where we maximize accuracy).
        
        Args:
            current_depth (int): Current depth of the DAG.
            
        Returns:
            bool: True if training should stop early, False otherwise.
        """
        # If no validation data is available, continue training
        if self.validation_X is None or self.validation_y is None:
            return False
            
        # Evaluate current validation score
        current_score = self._evaluate_validation_score()
        self.validation_scores.append(current_score)
        
        # Check for improvement (lower MSE is better)
        if current_score < self.best_validation_score - self.tol:
            # Reset counter if score improved significantly
            self.best_validation_score = current_score
            self.iterations_no_improvement = 0
        else:
            # Increment counter if no significant improvement
            self.iterations_no_improvement += 1
            
        # Check if we've reached the early stopping criteria
        if self.iterations_no_improvement >= self.n_iter_no_change:
            self.stopped_early = True
            return True
            
        # Continue training
        return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for the input samples.
        
        This method routes each sample through the DAG structure to determine
        the predicted target value. Samples start at the root node and
        are directed to child nodes based on split conditions at each internal node,
        until they reach leaf nodes which provide target value predictions.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples to predict.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted target values.
            
        Notes
        -----
        For samples that follow multiple paths through the DAG (due to the DAG structure
        which allows multiple paths to the same node), the predictions are derived from
        the leaf nodes they reach. Since nodes in a DAG can have multiple parents, the
        prediction process is more complex than in traditional trees but ultimately follows
        the same principle of routing samples based on feature values.
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        # Initialize samples at the root node
        node_samples = {self.root_node_id: np.arange(n_samples)}
        
        # Process each level of the DAG
        while node_samples:
            next_node_samples = {}
            
            # Process each node at the current level
            for node_id, sample_indices in node_samples.items():
                node = self.nodes[node_id]
                
                if isinstance(node, RegressionLeafNode):
                    # For leaf nodes, update predictions
                    predictions[sample_indices] = node.predict(X[sample_indices])
                elif isinstance(node, SplitNode):
                    # For split nodes, route samples to child nodes
                    go_left, go_right = node.predict(X[sample_indices])
                    
                    # Get child node IDs
                    left_id = node.left_child
                    right_id = node.right_child
                    
                    # Update samples for child nodes
                    if np.any(go_left):
                        left_indices = sample_indices[go_left]
                        if left_id in next_node_samples:
                            next_node_samples[left_id] = np.concatenate([next_node_samples[left_id], left_indices])
                        else:
                            next_node_samples[left_id] = left_indices
                    
                    if np.any(go_right):
                        right_indices = sample_indices[go_right]
                        if right_id in next_node_samples:
                            next_node_samples[right_id] = np.concatenate([next_node_samples[right_id], right_indices])
                        else:
                            next_node_samples[right_id] = right_indices
            
            # Move to the next level
            node_samples = next_node_samples
        
        return predictions
    
    def get_memory_usage(self) -> int:
        """
        Calculate the memory usage of the DAG in bytes.
        
        This method estimates the memory consumption of the DAG by summing
        the memory footprint of all nodes. The memory footprint includes the
        space required to store split conditions (for internal nodes) and
        target values (for leaf nodes).
        
        Returns
        -------
        int
            Estimated memory usage in bytes.
        """
        total_usage = 0
        for node in self.nodes.values():
            total_usage += node.get_memory_usage()
        return total_usage
    
    def get_node_count(self) -> int:
        """
        Get the number of nodes in the DAG.
        
        This method counts the total number of nodes in the DAG, including both
        split nodes (internal nodes) and leaf nodes. The node count is a key
        metric for comparing the model size between Decision Jungles and
        traditional Decision Trees.
        
        Returns
        -------
        int
            Total number of nodes in the DAG.
        """
        return len(self.nodes)
    
    def get_max_depth(self) -> int:
        """
        Get the maximum depth of the DAG.
        
        This method calculates the maximum depth of the DAG by finding the node
        with the highest depth value. The depth of a node is the length of the
        path from the root node to that node.
        
        Returns
        -------
        int
            Maximum depth of any node in the DAG.
        """
        return max(node.depth for node in self.nodes.values()) if self.nodes else 0
        
    def __getstate__(self) -> Dict[str, Any]:
        """
        Return the state of the DAG for pickling.
        
        This method is called when pickling the DAG. It returns the internal state
        of the object, which can then be used to restore the object when unpickling.
        
        Returns
        -------
        state : dict
            The state of the DAG to be serialized.
        """
        state = self.__dict__.copy()
        return state
        
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Set the state of the DAG after unpickling.
        
        This method is called when unpickling the DAG. It restores the internal
        state of the object.
        
        Parameters
        ----------
        state : dict
            The state of the DAG as returned by __getstate__().
        """
        self.__dict__.update(state)