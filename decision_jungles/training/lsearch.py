"""
Implementation of the LSearch optimization algorithm for Decision Jungles.

The LSearch algorithm optimizes both the split functions and branching structure
of the DAG by alternating between split-optimization and branch-optimization
steps as described in "Decision Jungles: Compact and Rich Models for Classification".
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set, Any
import random
from .objective import weighted_entropy_sum, optimize_split


class LSearch:
    """
    Implementation of the LSearch optimization algorithm for training Decision Jungles.
    
    The LSearch method alternates between optimizing split parameters and 
    optimizing branch assignments to minimize an objective function.
    """
    
    def __init__(self, n_classes: int, max_features: Optional[Union[int, float, str]] = "sqrt",
                random_state: Optional[int] = None):
        """
        Initialize a new LSearch optimizer.
        
        Args:
            n_classes (int): Number of classes in the dataset.
            max_features (int, float, str, optional): Number of features to consider
                for each split. If int, consider that many features. If float, consider
                that fraction of features. If "sqrt", consider sqrt(n_features) features.
                If "log2", consider log2(n_features) features. Default is "sqrt".
            random_state (int, optional): Random seed for reproducibility.
        """
        self.n_classes = n_classes
        self.max_features = max_features
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
    
    def _get_feature_indices(self, n_features: int) -> List[int]:
        """
        Get a subset of feature indices to consider for splitting.
        
        Args:
            n_features (int): Total number of features.
            
        Returns:
            List[int]: Indices of features to consider.
        """
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_feats = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                max_feats = max(1, int(np.log2(n_features)))
            else:
                raise ValueError(f"Invalid max_features value: {self.max_features}")
        elif isinstance(self.max_features, float):
            max_feats = max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            max_feats = min(max_feats, n_features)
        else:
            raise ValueError(f"Invalid max_features value: {self.max_features}")
        
        # Randomly select feature indices
        indices = list(range(n_features))
        self._rng.shuffle(indices)
        return indices[:max_feats]
    
    def optimize(self, X: np.ndarray, y: np.ndarray, parent_nodes: List[int], 
                node_samples: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                n_child_nodes: int, max_iterations: int = 100,
                is_categorical: Optional[np.ndarray] = None,
                feature_bins: Optional[Dict[int, Dict[Any, int]]] = None) -> Tuple[Dict[int, Tuple[int, float, bool, Optional[Set[Any]]]], Dict[int, int], Dict[int, int]]:
        """
        Optimize one level of the DAG using the LSearch algorithm.
        
        Args:
            X (np.ndarray): Feature matrix of all samples.
            y (np.ndarray): Class labels of all samples.
            parent_nodes (List[int]): List of parent node IDs.
            node_samples (Dict[int, Tuple[np.ndarray, np.ndarray]]): Dictionary mapping each 
                parent node ID to a tuple of (indices, class_labels) of samples at that node.
            n_child_nodes (int): Number of child nodes to create.
            max_iterations (int): Maximum number of optimization iterations.
            is_categorical (np.ndarray, optional): Boolean mask indicating which features are categorical.
            feature_bins (Dict, optional): Dictionary mapping feature indices to category-to-bin mappings.
            
        Returns:
            Tuple[Dict[int, Tuple[int, float, bool, Optional[Set]]], Dict[int, int], Dict[int, int]]:
                1. Split parameters (feature_idx, threshold, is_categorical, categories_left) for each parent node.
                2. Left child assignments for each parent node.
                3. Right child assignments for each parent node.
        """
        n_features = X.shape[1]
        n_parent_nodes = len(parent_nodes)
        
        # Initialize split parameters randomly for each parent node
        split_params = {}
        for node_id in parent_nodes:
            indices, labels = node_samples[node_id]
            if len(indices) > 1:
                # Randomly select a feature
                feature_idx = self._rng.randint(0, n_features)
                
                # Check if the feature is categorical
                is_cat = (is_categorical is not None and 
                          feature_idx < len(is_categorical) and 
                          is_categorical[feature_idx] and
                          feature_bins is not None and
                          feature_idx in feature_bins)
                
                feature_values = X[indices, feature_idx]
                if is_cat:
                    # For categorical features, randomly divide categories into left and right
                    categories = list(feature_bins[feature_idx].keys())
                    if len(categories) > 1:
                        # Randomly select categories to go left
                        num_left = self._rng.randint(1, len(categories))
                        self._rng.shuffle(categories)
                        left_cats = set(categories[:num_left])
                        split_params[node_id] = (feature_idx, 0.0, True, left_cats)
                    else:
                        # Default split for categorical features with only one category
                        split_params[node_id] = (feature_idx, 0.0, False, None)
                else:
                    # For numerical features, use the original approach
                    if len(np.unique(feature_values)) > 1:
                        non_nan_values = feature_values[~np.isnan(feature_values)]
                        if len(non_nan_values) > 0:
                            min_val, max_val = np.min(non_nan_values), np.max(non_nan_values)
                            threshold = self._rng.uniform(min_val, max_val)
                            split_params[node_id] = (feature_idx, threshold, False, None)
                        else:
                            # If all values are NaN, use a default split
                            split_params[node_id] = (feature_idx, 0.0, False, None)
                    else:
                        # If only one unique value, use a default split
                        split_params[node_id] = (feature_idx, feature_values[0], False, None)
            else:
                # For nodes with 0 or 1 sample, use a default split
                split_params[node_id] = (0, 0.0, False, None)
        
        # Initialize branch assignments greedily
        left_child = {}
        right_child = {}
        available_child_ids = list(range(n_child_nodes))
        
        # Sort parent nodes by decreasing sample size
        # Make sure node_samples contains the parent_node before accessing it
        sorted_parents = sorted(
            [node_id for node_id in parent_nodes if node_id in node_samples],
            key=lambda node_id: len(node_samples[node_id][0]),
            reverse=True
        )
        
        # Assign children to parents greedily
        for node_id in sorted_parents:
            if len(available_child_ids) >= 2:
                left_child[node_id] = available_child_ids.pop(0)
                right_child[node_id] = available_child_ids.pop(0)
            else:
                # If running out of available child IDs, reuse existing ones
                left_child[node_id] = 0 if not left_child else list(left_child.values())[0]
                right_child[node_id] = 0 if not right_child else list(right_child.values())[0]
        
        # Main optimization loop
        for iteration in range(max_iterations):
            changed = False
            
            # Split optimization step
            for node_id in parent_nodes:
                indices, labels = node_samples[node_id]
                if len(indices) <= 1:
                    continue  # Skip nodes with too few samples
                
                # Get a random subset of features to consider
                feature_indices = self._get_feature_indices(n_features)
                
                # Current split parameters
                current_feature, current_threshold, is_cat, categories_left = split_params[node_id]
                
                # Get current left/right sample indices
                if is_cat and categories_left:
                    # For categorical features, check if the value is in the categories_left set
                    left_mask = np.zeros(len(indices), dtype=bool)
                    for i in range(len(indices)):
                        idx = indices[i]
                        feature_value = X[idx, current_feature]
                        if not np.isnan(feature_value) and feature_value in categories_left:
                            left_mask[i] = True
                else:
                    # For numerical features
                    left_mask = X[indices, current_feature] <= current_threshold
                    # Handle NaN values for numerical features - send them to the right
                    nan_mask = np.isnan(X[indices, current_feature])
                    left_mask[nan_mask] = False
                
                right_mask = ~left_mask
                
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]
                
                # Get all samples for each child
                child_samples = [[] for _ in range(n_child_nodes)]
                
                for p_id in parent_nodes:
                    p_indices, p_labels = node_samples[p_id]
                    p_feature, p_threshold, p_is_cat, p_categories_left = split_params[p_id]
                    
                    if p_is_cat and p_categories_left:
                        # For categorical features
                        p_left_mask = np.zeros(len(p_indices), dtype=bool)
                        for i in range(len(p_indices)):
                            idx = p_indices[i]
                            feature_value = X[idx, p_feature]
                            if not np.isnan(feature_value) and feature_value in p_categories_left:
                                p_left_mask[i] = True
                    else:
                        # For numerical features
                        p_left_mask = X[p_indices, p_feature] <= p_threshold
                        # Handle NaN values - send them to the right
                        nan_mask = np.isnan(X[p_indices, p_feature])
                        p_left_mask[nan_mask] = False
                        
                    p_right_mask = ~p_left_mask
                    
                    left_child_id = left_child[p_id]
                    right_child_id = right_child[p_id]
                    
                    # Add samples to appropriate child node
                    if len(p_indices[p_left_mask]) > 0:
                        child_samples[left_child_id].extend(p_labels[p_left_mask])
                    
                    if len(p_indices[p_right_mask]) > 0:
                        child_samples[right_child_id].extend(p_labels[p_right_mask])
                
                # Convert lists to numpy arrays
                child_samples = [np.array(s) if s else np.array([], dtype=int) for s in child_samples]
                
                # Calculate current objective value
                current_obj = weighted_entropy_sum(child_samples, self.n_classes)
                
                # Try to find a better split, now with categorical feature support
                best_feature_idx, best_threshold, _, best_is_cat, best_cats_left = optimize_split(
                    X[indices], labels, feature_indices, self.n_classes,
                    is_categorical=is_categorical, feature_bins=feature_bins
                )
                
                if best_feature_idx >= 0:
                    # Get new left/right sample indices
                    if best_is_cat and best_cats_left:
                        # For categorical features
                        new_left_mask = np.zeros(len(indices), dtype=bool)
                        for i in range(len(indices)):
                            idx = indices[i]
                            feature_value = X[idx, best_feature_idx]
                            if not np.isnan(feature_value) and feature_value in best_cats_left:
                                new_left_mask[i] = True
                    else:
                        # For numerical features
                        new_left_mask = X[indices, best_feature_idx] <= best_threshold
                        # Handle NaN values - send them to the right
                        nan_mask = np.isnan(X[indices, best_feature_idx])
                        new_left_mask[nan_mask] = False
                        
                    new_right_mask = ~new_left_mask
                    
                    # Update the child samples with new split
                    new_child_samples = [list(s) for s in child_samples]
                    
                    # Remove old samples from children
                    if len(left_indices) > 0:
                        for i in range(len(left_indices)):
                            if left_indices[i] < len(y):  # Safety check
                                try:
                                    new_child_samples[left_child[node_id]].remove(y[left_indices[i]])
                                except ValueError:
                                    pass  # Sample might not be in the list
                    
                    if len(right_indices) > 0:
                        for i in range(len(right_indices)):
                            if right_indices[i] < len(y):  # Safety check
                                try:
                                    new_child_samples[right_child[node_id]].remove(y[right_indices[i]])
                                except ValueError:
                                    pass  # Sample might not be in the list
                    
                    # Add new samples to children
                    new_left_indices = indices[new_left_mask]
                    new_right_indices = indices[new_right_mask]
                    
                    if len(new_left_indices) > 0:
                        new_child_samples[left_child[node_id]].extend(y[new_left_indices])
                    
                    if len(new_right_indices) > 0:
                        new_child_samples[right_child[node_id]].extend(y[new_right_indices])
                    
                    # Convert lists to numpy arrays
                    new_child_samples = [np.array(s) if s else np.array([], dtype=int) for s in new_child_samples]
                    
                    # Calculate new objective value
                    new_obj = weighted_entropy_sum(new_child_samples, self.n_classes)
                    
                    # If better, update split parameters
                    if new_obj < current_obj:
                        split_params[node_id] = (best_feature_idx, best_threshold, best_is_cat, best_cats_left)
                        changed = True
            
            # Branch optimization step
            for node_id in parent_nodes:
                indices, labels = node_samples[node_id]
                if len(indices) <= 1:
                    continue
                
                feature_idx, threshold, is_cat, categories_left = split_params[node_id]
                
                # Get left/right sample indices and labels
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
                
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]
                left_labels = labels[left_mask]
                right_labels = labels[right_mask]
                
                # Try redirecting left branch
                current_left_child = left_child[node_id]
                
                for new_left_child in range(n_child_nodes):
                    if new_left_child == current_left_child:
                        continue  # Skip current assignment
                    
                    # Get all samples for each child with new left child assignment
                    new_child_samples = [[] for _ in range(n_child_nodes)]
                    
                    for p_id in parent_nodes:
                        p_indices, p_labels = node_samples[p_id]
                        p_feature, p_threshold, p_is_cat, p_categories_left = split_params[p_id]
                        
                        if p_is_cat and p_categories_left:
                            # For categorical features
                            p_left_mask = np.zeros(len(p_indices), dtype=bool)
                            for i in range(len(p_indices)):
                                idx = p_indices[i]
                                feature_value = X[idx, p_feature]
                                if not np.isnan(feature_value) and feature_value in p_categories_left:
                                    p_left_mask[i] = True
                        else:
                            # For numerical features
                            p_left_mask = X[p_indices, p_feature] <= p_threshold
                            # Handle NaN values - send them to the right
                            nan_mask = np.isnan(X[p_indices, p_feature])
                            p_left_mask[nan_mask] = False
                            
                        p_right_mask = ~p_left_mask
                        
                        # Use new left child for current node, otherwise use existing assignments
                        p_left_child_id = new_left_child if p_id == node_id else left_child[p_id]
                        p_right_child_id = right_child[p_id]
                        
                        # Add samples to appropriate child node
                        if len(p_indices[p_left_mask]) > 0:
                            new_child_samples[p_left_child_id].extend(p_labels[p_left_mask])
                        
                        if len(p_indices[p_right_mask]) > 0:
                            new_child_samples[p_right_child_id].extend(p_labels[p_right_mask])
                    
                    # Convert lists to numpy arrays
                    new_child_samples = [np.array(s) if s else np.array([], dtype=int) for s in new_child_samples]
                    
                    # Calculate objective value with new assignment
                    new_obj = weighted_entropy_sum(new_child_samples, self.n_classes)
                    
                    # Get all samples for each child with current assignments
                    current_child_samples = [[] for _ in range(n_child_nodes)]
                    
                    for p_id in parent_nodes:
                        p_indices, p_labels = node_samples[p_id]
                        p_feature, p_threshold = split_params[p_id]
                        
                        p_left_mask = X[p_indices, p_feature] <= p_threshold
                        p_right_mask = ~p_left_mask
                        
                        # Use current assignments
                        p_left_child_id = left_child[p_id]
                        p_right_child_id = right_child[p_id]
                        
                        # Add samples to appropriate child node
                        if len(p_indices[p_left_mask]) > 0:
                            current_child_samples[p_left_child_id].extend(p_labels[p_left_mask])
                        
                        if len(p_indices[p_right_mask]) > 0:
                            current_child_samples[p_right_child_id].extend(p_labels[p_right_mask])
                    
                    # Convert lists to numpy arrays
                    current_child_samples = [np.array(s) if s else np.array([], dtype=int) for s in current_child_samples]
                    
                    # Calculate current objective value
                    current_obj = weighted_entropy_sum(current_child_samples, self.n_classes)
                    
                    # If better, update left child assignment
                    if new_obj < current_obj:
                        left_child[node_id] = new_left_child
                        changed = True
                        break  # Stop after first improvement
                
                # Try redirecting right branch
                current_right_child = right_child[node_id]
                
                for new_right_child in range(n_child_nodes):
                    if new_right_child == current_right_child:
                        continue  # Skip current assignment
                    
                    # Get all samples for each child with new right child assignment
                    new_child_samples = [[] for _ in range(n_child_nodes)]
                    
                    for p_id in parent_nodes:
                        p_indices, p_labels = node_samples[p_id]
                        p_feature, p_threshold = split_params[p_id]
                        
                        p_left_mask = X[p_indices, p_feature] <= p_threshold
                        p_right_mask = ~p_left_mask
                        
                        # Use new right child for current node, otherwise use existing assignments
                        p_left_child_id = left_child[p_id]
                        p_right_child_id = new_right_child if p_id == node_id else right_child[p_id]
                        
                        # Add samples to appropriate child node
                        if len(p_indices[p_left_mask]) > 0:
                            new_child_samples[p_left_child_id].extend(p_labels[p_left_mask])
                        
                        if len(p_indices[p_right_mask]) > 0:
                            new_child_samples[p_right_child_id].extend(p_labels[p_right_mask])
                    
                    # Convert lists to numpy arrays
                    new_child_samples = [np.array(s) if s else np.array([], dtype=int) for s in new_child_samples]
                    
                    # Calculate objective value with new assignment
                    new_obj = weighted_entropy_sum(new_child_samples, self.n_classes)
                    
                    # Get all samples for each child with current assignments
                    current_child_samples = [[] for _ in range(n_child_nodes)]
                    
                    for p_id in parent_nodes:
                        p_indices, p_labels = node_samples[p_id]
                        p_feature, p_threshold = split_params[p_id]
                        
                        p_left_mask = X[p_indices, p_feature] <= p_threshold
                        p_right_mask = ~p_left_mask
                        
                        # Use current assignments
                        p_left_child_id = left_child[p_id]
                        p_right_child_id = right_child[p_id]
                        
                        # Add samples to appropriate child node
                        if len(p_indices[p_left_mask]) > 0:
                            current_child_samples[p_left_child_id].extend(p_labels[p_left_mask])
                        
                        if len(p_indices[p_right_mask]) > 0:
                            current_child_samples[p_right_child_id].extend(p_labels[p_right_mask])
                    
                    # Convert lists to numpy arrays
                    current_child_samples = [np.array(s) if s else np.array([], dtype=int) for s in current_child_samples]
                    
                    # Calculate current objective value
                    current_obj = weighted_entropy_sum(current_child_samples, self.n_classes)
                    
                    # If better, update right child assignment
                    if new_obj < current_obj:
                        right_child[node_id] = new_right_child
                        changed = True
                        break  # Stop after first improvement
            
            # If no changes made in this iteration, we've converged
            if not changed:
                break
        
        return split_params, left_child, right_child
