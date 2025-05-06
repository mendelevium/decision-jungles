"""
Directed Acyclic Graph (DAG) implementation for Decision Jungles.

This module implements the DAG class that represents a single directed acyclic
graph within a Decision Jungle ensemble.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set, Any, cast
import heapq
from .node import Node, SplitNode, LeafNode
from .training.lsearch import LSearch
try:
    from .training.optimized_lsearch import OptimizedLSearch
    OPTIMIZED_LSEARCH_AVAILABLE = True
except ImportError:
    OPTIMIZED_LSEARCH_AVAILABLE = False
try:
    from .training.cyth_lsearch import CythLSearch
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
from .training.clustersearch import ClusterSearch


class DAG:
    """
    A Directed Acyclic Graph (DAG) for Decision Jungle classification.
    
    This class implements a single DAG component of a Decision Jungle ensemble.
    Unlike a traditional decision tree where each node has exactly one parent,
    nodes in a DAG can have multiple parents, allowing for a more compact
    representation while maintaining or improving predictive performance.
    
    The DAG is built level-by-level with a controlled width at each level,
    determined by the merging schedule. During training, the LSearch or
    ClusterSearch optimization algorithm is used to determine both the optimal
    split parameters and the optimal branch assignments.
    
    Attributes
    ----------
    nodes : Dict[int, Node]
        Dictionary of all nodes in the DAG, keyed by node ID.
    
    root_node_id : int
        ID of the root node of the DAG.
    
    n_classes : int
        Number of classes in the classification problem.
        
    next_node_id : int
        Counter for generating unique node IDs.
        
    max_width : int
        Maximum width of each level (M parameter from the paper).
        
    max_depth : int or None
        Maximum depth of the DAG. If None, the DAG will grow until all
        leaves are pure or contain less than min_samples_split samples.
    
    See Also
    --------
    SplitNode : Internal nodes with splitting functionality.
    LeafNode : Terminal nodes with class distributions.
    LSearch : Main optimization algorithm for DAG construction.
    ClusterSearch : Alternative optimization algorithm.
    """
    
    def __init__(self, n_classes: int, max_depth: Optional[int] = None,
                min_samples_split: int = 2, min_samples_leaf: int = 1,
                min_impurity_decrease: float = 0.0, max_features: Optional[Union[int, float, str]] = "sqrt",
                random_state: Optional[int] = None, merging_schedule: str = "exponential",
                max_width: int = 256, optimization_method: str = "lsearch",
                use_optimized: bool = True, early_stopping: bool = False,
                validation_X: Optional[np.ndarray] = None, validation_y: Optional[np.ndarray] = None,
                n_iter_no_change: int = 5, tol: float = 1e-4,
                is_categorical: Optional[np.ndarray] = None, 
                feature_bins: Optional[Dict[int, Dict[Any, int]]] = None):
        """
        Initialize a new DAG.
        
        Args:
            n_classes (int): Number of classes in the classification problem.
            max_depth (int, optional): Maximum depth of the DAG.
            min_samples_split (int): Minimum number of samples required to split a node.
            min_samples_leaf (int): Minimum number of samples required at a leaf node.
            min_impurity_decrease (float): Minimum impurity decrease required for a split.
            max_features (int, float, str, optional): Number of features to consider for splits.
            random_state (int, optional): Random seed for reproducibility.
            merging_schedule (str): Type of merging schedule to use ("constant", "exponential", "kinect").
            max_width (int): Maximum width of each level (M parameter).
            optimization_method (str): Method for DAG optimization ("lsearch" or "clustersearch").
            use_optimized (bool): Whether to use the optimized implementation of the LSearch algorithm.
            early_stopping (bool): Whether to use early stopping to terminate training when validation score is not improving.
            validation_X (ndarray, optional): Validation features for early stopping.
            validation_y (ndarray, optional): Validation labels for early stopping.
            n_iter_no_change (int): Number of iterations with no improvement to wait before early stopping.
            tol (float): Tolerance for the early stopping criterion.
            is_categorical (ndarray, optional): Boolean mask indicating which features are categorical.
            feature_bins (Dict, optional): Dictionary mapping feature indices to category-to-bin mappings.
        """
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
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
        self.best_validation_score = -np.inf
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
    
    def _get_optimizer(self):
        """
        Get the appropriate optimization algorithm based on configuration.
        
        Returns:
            Object: The optimization algorithm instance.
        """
        if self.optimization_method.lower() == "lsearch":
            # Check if Cython version is available
            if self.use_optimized and CYTHON_AVAILABLE:
                return CythLSearch(self.n_classes)
            # If Cython not available, use optimized version if available
            elif self.use_optimized and OPTIMIZED_LSEARCH_AVAILABLE:
                return OptimizedLSearch(self.n_classes, self.max_features, self.random_state)
            # Fallback to standard implementation
            else:
                return LSearch(self.n_classes, self.max_features, self.random_state)
        elif self.optimization_method.lower() == "clustersearch":
            return ClusterSearch(self.n_classes, self.max_features, self.random_state)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the DAG to the training data.
        
        This method builds a directed acyclic graph (DAG) structure for classification
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
            The target values (class labels).
            
        Notes
        -----
        The fitting process involves these key steps:
        
        1. Create a root node containing all training samples
        2. For each level up to max_depth:
           a. Determine the width of the next level based on the merging schedule
           b. Create temporary nodes for the next level
           c. Use the optimization algorithm (LSearch or ClusterSearch) to find:
              - Optimal split parameters for each parent node
              - Optimal branch assignments to child nodes
           d. Update the DAG structure with the optimized splits and branches
        3. Stop when max_depth is reached or no more nodes can be split
        
        The DAG is constructed to maximize information gain while respecting
        the width constraints, leading to a more memory-efficient model.
        """
        # Create the root node
        self.root_node_id = self._get_node_id()
        self.nodes[self.root_node_id] = LeafNode(self.root_node_id, 0, self.n_classes)
        self.nodes[self.root_node_id].update_distribution(y)
        
        # Track samples at each node
        node_samples = {self.root_node_id: (np.arange(X.shape[0]), y)}
        
        # Track leaf nodes to process
        leaf_nodes = [self.root_node_id]
        current_depth = 0
        
        # Initialize the optimizer
        optimizer = self._get_optimizer()
        
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
                    
                indices, labels = node_samples[node_id]
                # Skip if there are no samples or not enough samples to split
                if (len(indices) == 0 or 
                    len(indices) < self.min_samples_split or
                    len(np.unique(labels)) <= 1):
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
                self.nodes[node_id] = LeafNode(node_id, next_depth, self.n_classes)
                next_level_nodes.append(node_id)
            
            # Optimize splits and branch assignments using the selected optimizer
            split_params, left_child, right_child = optimizer.optimize(
                X, y, splittable_nodes, node_samples, next_level_width,
                is_categorical=self.is_categorical, feature_bins=self.feature_bins
            )
            
            # Update node structure based on the optimization results
            new_node_samples = {node_id: (np.array([], dtype=int), np.array([], dtype=int))
                               for node_id in next_level_nodes}
            
            # Process splittable nodes
            for node_id in splittable_nodes:
                indices, labels = node_samples[node_id]
                
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
                        left_labels = labels[left_mask]
                        
                        # Append to existing samples for this child node
                        existing_indices, existing_labels = new_node_samples[left_id]
                        if len(existing_indices) == 0:
                            new_node_samples[left_id] = (left_indices, left_labels)
                        else:
                            new_node_samples[left_id] = (
                                np.concatenate([existing_indices, left_indices]),
                                np.concatenate([existing_labels, left_labels])
                            )
                        
                        # Update class distribution
                        self.nodes[left_id].update_distribution(left_labels)
                    
                    if np.any(right_mask):
                        right_indices = indices[right_mask]
                        right_labels = labels[right_mask]
                        
                        # Append to existing samples for this child node
                        existing_indices, existing_labels = new_node_samples[right_id]
                        if len(existing_indices) == 0:
                            new_node_samples[right_id] = (right_indices, right_labels)
                        else:
                            new_node_samples[right_id] = (
                                np.concatenate([existing_indices, right_indices]),
                                np.concatenate([existing_labels, right_labels])
                            )
                        
                        # Update class distribution
                        self.nodes[right_id].update_distribution(right_labels)
            
            # Process non-splittable leaf nodes at current depth
            non_splittable = [node_id for node_id in leaf_nodes if node_id not in splittable_nodes]
            
            # Keep non-splittable leaf nodes as they are
            for node_id in non_splittable:
                pass  # Already a leaf node, no need to modify
            
            # Update leaf nodes for the next iteration
            leaf_nodes = [node_id for node_id in next_level_nodes
                          if isinstance(self.nodes[node_id], LeafNode)]
            
            # Update node samples for the next iteration
            # Include all leaf nodes, using empty arrays for those without samples
            # This prevents KeyError when accessing node_samples in subsequent iterations
            node_samples = {}
            for node_id in leaf_nodes:
                if node_id in new_node_samples and len(new_node_samples[node_id][0]) > 0:
                    node_samples[node_id] = new_node_samples[node_id]
                else:
                    # Include the node with empty sample arrays
                    node_samples[node_id] = (np.array([], dtype=int), np.array([], dtype=int))
            
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
            float: Validation accuracy score, or -inf if no validation data is available.
        """
        if self.validation_X is None or self.validation_y is None:
            return -np.inf
            
        # Make predictions on validation set
        y_pred = self.predict(self.validation_X)
        
        # Calculate accuracy
        correct = np.sum(y_pred == self.validation_y)
        total = len(self.validation_y)
        
        return correct / total if total > 0 else 0.0
    
    def _check_early_stopping(self, current_depth: int) -> bool:
        """
        Check if early stopping criteria are met.
        
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
        
        # Check for improvement
        if current_score > self.best_validation_score + self.tol:
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
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input samples.
        
        This method routes each sample through the DAG structure to determine
        the probability of each class. Samples start at the root node and
        are directed to child nodes based on split conditions at each internal node,
        until they reach leaf nodes which provide class probability distributions.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples to predict.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities for each sample. The order of classes
            corresponds to the internal ordering of classes in the DAG.
            
        Notes
        -----
        For samples that follow multiple paths through the DAG (due to the DAG structure
        which allows multiple paths to the same node), the probability distributions are
        derived from the leaf nodes they reach. Since nodes in a DAG can have multiple
        parents, the prediction process is more complex than in traditional trees but
        ultimately follows the same principle of routing samples based on feature values.
        """
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes))
        
        # Initialize samples at the root node
        node_samples = {self.root_node_id: np.arange(n_samples)}
        
        # Process each level of the DAG
        while node_samples:
            next_node_samples = {}
            
            # Process each node at the current level
            for node_id, sample_indices in node_samples.items():
                node = self.nodes[node_id]
                
                if isinstance(node, LeafNode):
                    # For leaf nodes, update probabilities
                    proba[sample_indices] = node.predict(X[sample_indices])
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
        
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input samples.
        
        This method predicts the most likely class for each input sample by finding
        the class with the highest probability from predict_proba.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples to predict.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted class labels. Each value corresponds to the class with
            the highest probability for that sample.
            
        Notes
        -----
        This is a convenience method that calls predict_proba and then takes the
        argmax of the probability distributions to determine the most likely class.
        For the full probability distribution across all classes, use predict_proba.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_memory_usage(self) -> int:
        """
        Calculate the memory usage of the DAG in bytes.
        
        This method estimates the memory consumption of the DAG by summing
        the memory footprint of all nodes. The memory footprint includes the
        space required to store split conditions (for internal nodes) and
        class distributions (for leaf nodes).
        
        Returns
        -------
        int
            Estimated memory usage in bytes.
            
        Notes
        -----
        Memory efficiency is one of the key advantages of Decision Jungles over
        traditional Decision Trees/Forests. By allowing nodes to have multiple
        parents, the DAG structure can represent complex decision boundaries
        with fewer nodes, resulting in reduced memory consumption.
        
        The actual memory usage in a running system may vary due to implementation
        details, garbage collection, and memory alignment requirements of the
        programming language and runtime environment.
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
            
        Notes
        -----
        The node count is directly related to the memory footprint of the model.
        One of the main advantages of Decision Jungles is achieving similar or
        better predictive performance with significantly fewer nodes compared
        to Decision Trees, due to the node merging process which allows multiple
        parent nodes to share the same child nodes.
        
        When comparing with Decision Forests, the ratio of node counts (Forest/Jungle)
        provides a good measure of the memory savings achieved by the jungle.
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
            
        Notes
        -----
        In Decision Jungles, the depth of a node is well-defined even though nodes
        can have multiple parents, because the DAG is constructed level-by-level.
        All nodes at a given level have the same depth, which simplifies the
        analysis of the model's complexity.
        
        The maximum depth affects both the model's capacity to represent complex
        patterns and its computational complexity during prediction, as samples
        must traverse at most this many levels to reach a leaf node.
        """
        return max(node.depth for node in self.nodes.values()) if self.nodes else 0
        
    def get_decision_path(self, sample: np.ndarray) -> List[int]:
        """
        Get the decision path for a single sample through the DAG.
        
        This method traces the path a sample takes from the root node to a leaf node
        in the DAG. Since the DAG allows multiple paths to the same node, this method
        follows the first valid path it finds when multiple options exist.
        
        Parameters
        ----------
        sample : ndarray of shape (n_features,)
            The single sample to trace through the DAG.
            
        Returns
        -------
        path : List[int]
            A list of node IDs representing the path from root to leaf.
            
        Notes
        -----
        This method is useful for model interpretability, allowing visualization of
        the specific decision path for a given input sample. In a DAG, there could
        potentially be multiple valid paths to the same leaf node, but this method
        returns the first one it discovers.
        
        Example
        -------
        >>> path = dag.get_decision_path(X[0])
        >>> print(f"Sample follows path: {path}")
        """
        path = []
        current_node_id = self.root_node_id
        
        # Handle the case of no nodes
        if not self.nodes or current_node_id is None:
            return path
            
        # Add the root node to the path
        path.append(current_node_id)
        
        # Traverse the DAG until we reach a leaf node
        while True:
            current_node = self.nodes[current_node_id]
            
            # If we've reached a leaf node, we're done
            if isinstance(current_node, LeafNode):
                break
                
            # For split nodes, determine which child to go to
            if isinstance(current_node, SplitNode):
                # Get the feature value
                feature_idx = current_node.feature_idx
                feature_value = sample[feature_idx]
                
                # Determine whether to go left or right
                if current_node.is_categorical and current_node.categories_left:
                    # Categorical split
                    if (not np.isnan(feature_value) and 
                        feature_value in current_node.categories_left):
                        # Go left
                        current_node_id = current_node.left_child
                    else:
                        # Go right
                        current_node_id = current_node.right_child
                else:
                    # Numerical split
                    if np.isnan(feature_value):
                        # NaN values go right
                        current_node_id = current_node.right_child
                    elif feature_value <= current_node.threshold:
                        # Go left
                        current_node_id = current_node.left_child
                    else:
                        # Go right
                        current_node_id = current_node.right_child
                
                # Add the next node to the path
                path.append(current_node_id)
            else:
                # This should never happen, but just in case
                break
                
        return path
        
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
        # Remove the optimizer reference since it's not needed for prediction
        # and will be recreated if needed
        if '_optimizer' in state:
            del state['_optimizer']
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
