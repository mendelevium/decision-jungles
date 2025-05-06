"""
Implementation of the ClusterSearch optimization algorithm for Decision Jungles.

The ClusterSearch algorithm optimizes the structure of the DAG by first building
temporary child nodes in a tree-like manner and then clustering them based on
similarity in class distributions as described in "Decision Jungles: Compact and 
Rich Models for Classification".
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set, Any, cast
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from .objective import weighted_entropy_sum, optimize_split


class ClusterSearch:
    """
    Implementation of the ClusterSearch optimization algorithm for training Decision Jungles.
    
    The ClusterSearch method first builds temporary child nodes via conventional 
    tree-based training procedures, then clusters these nodes to produce a DAG.
    """
    
    def __init__(self, n_classes: int, max_features: Optional[Union[int, float, str]] = "sqrt",
                random_state: Optional[int] = None):
        """
        Initialize a new ClusterSearch optimizer.
        
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
            max_feats = min(self.max_features, n_features)
        else:
            raise ValueError(f"Invalid max_features value: {self.max_features}")
        
        # Randomly select feature indices
        indices = list(range(n_features))
        self._rng.shuffle(indices)
        return indices[:max_feats]
    
    def _build_temporary_nodes(self, X: np.ndarray, y: np.ndarray, 
                             parent_nodes: List[int],
                             node_samples: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Dict[str, Any]]:
        """
        Build temporary child nodes using conventional tree-based training.
        
        Args:
            X (np.ndarray): Feature matrix of all samples.
            y (np.ndarray): Class labels of all samples.
            parent_nodes (List[int]): List of parent node IDs.
            node_samples (Dict[int, Tuple[np.ndarray, np.ndarray]]): Dictionary mapping
                each parent node ID to a tuple of (indices, class_labels) of samples.
                
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping temporary node IDs to their attributes.
        """
        n_features = X.shape[1]
        temp_nodes = {}
        next_temp_id = 0
        
        for node_id in parent_nodes:
            indices, labels = node_samples[node_id]
            
            if len(indices) <= 1 or len(np.unique(labels)) <= 1:
                continue  # Skip nodes with too few samples or pure class distribution
            
            # Get a random subset of features to consider
            feature_indices = self._get_feature_indices(n_features)
            
            # Find the best split
            best_feature_idx, best_threshold, best_info_gain = optimize_split(
                X[indices], labels, feature_indices, self.n_classes
            )
            
            if best_feature_idx >= 0 and best_info_gain > 0:
                # Create temporary left and right child nodes
                left_mask = X[indices, best_feature_idx] <= best_threshold
                right_mask = ~left_mask
                
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]
                left_labels = labels[left_mask]
                right_labels = labels[right_mask]
                
                # Store left child
                if len(left_indices) > 0:
                    left_id = next_temp_id
                    next_temp_id += 1
                    
                    # Calculate class distribution
                    class_distribution = np.zeros(self.n_classes, dtype=int)
                    for label in left_labels:
                        class_distribution[label] += 1
                    
                    # Normalize to get probabilities
                    class_probs = class_distribution / len(left_labels) if len(left_labels) > 0 else np.zeros(self.n_classes)
                    
                    temp_nodes[left_id] = {
                        'parent': node_id,
                        'is_left_child': True,
                        'indices': left_indices,
                        'labels': left_labels,
                        'class_distribution': class_distribution,
                        'class_probs': class_probs,
                        'n_samples': len(left_indices)
                    }
                
                # Store right child
                if len(right_indices) > 0:
                    right_id = next_temp_id
                    next_temp_id += 1
                    
                    # Calculate class distribution
                    class_distribution = np.zeros(self.n_classes, dtype=int)
                    for label in right_labels:
                        class_distribution[label] += 1
                    
                    # Normalize to get probabilities
                    class_probs = class_distribution / len(right_labels) if len(right_labels) > 0 else np.zeros(self.n_classes)
                    
                    temp_nodes[right_id] = {
                        'parent': node_id,
                        'is_left_child': False,
                        'indices': right_indices,
                        'labels': right_labels,
                        'class_distribution': class_distribution,
                        'class_probs': class_probs,
                        'n_samples': len(right_indices)
                    }
        
        return temp_nodes
    
    def _cluster_nodes(self, temp_nodes: Dict[int, Dict[str, Any]], n_clusters: int) -> Dict[int, int]:
        """
        Cluster the temporary nodes based on their class distributions.
        
        Args:
            temp_nodes (Dict[int, Dict[str, Any]]): Dictionary of temporary nodes.
            n_clusters (int): Number of clusters to create.
            
        Returns:
            Dict[int, int]: Mapping from temporary node ID to cluster ID.
        """
        if len(temp_nodes) <= n_clusters:
            # If we have fewer nodes than clusters, each node is its own cluster
            return {node_id: i for i, node_id in enumerate(temp_nodes.keys())}
        
        # Extract class probability distributions from each node
        node_ids = list(temp_nodes.keys())
        distributions = np.array([temp_nodes[node_id]['class_probs'] for node_id in node_ids])
        
        # Calculate pairwise distances between distributions using Jensen-Shannon divergence
        # For simplicity, we use Euclidean distance here, but JS divergence would be better
        distances = pdist(distributions, metric='euclidean')
        
        # Perform hierarchical clustering
        Z = linkage(distances, method='ward')
        
        # Determine the cluster assignments
        cluster_assignments = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Create mapping from node ID to cluster ID
        node_to_cluster = {node_id: cluster_id - 1 for node_id, cluster_id in zip(node_ids, cluster_assignments)}
        
        return node_to_cluster
    
    def optimize(self, X: np.ndarray, y: np.ndarray, parent_nodes: List[int], 
                node_samples: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                n_child_nodes: int, max_iterations: int = 1,
                is_categorical: Optional[np.ndarray] = None,
                feature_bins: Optional[Dict[int, Dict[Any, int]]] = None) -> Tuple[Dict[int, Tuple[int, float, bool, Optional[Set[Any]]]], Dict[int, int], Dict[int, int]]:
        """
        Optimize one level of the DAG using the ClusterSearch algorithm.
        
        Args:
            X (np.ndarray): Feature matrix of all samples.
            y (np.ndarray): Class labels of all samples.
            parent_nodes (List[int]): List of parent node IDs.
            node_samples (Dict[int, Tuple[np.ndarray, np.ndarray]]): Dictionary mapping each 
                parent node ID to a tuple of (indices, class_labels) of samples at that node.
            n_child_nodes (int): Number of child nodes to create.
            max_iterations (int): Maximum number of optimization iterations.
                For ClusterSearch, this is typically 1.
            
        Returns:
            Tuple[Dict[int, Tuple[int, float]], Dict[int, int], Dict[int, int]]:
                1. Split parameters (feature_idx, threshold) for each parent node.
                2. Left child assignments for each parent node.
                3. Right child assignments for each parent node.
        """
        n_features = X.shape[1]
        
        # Split optimization step - find the best split for each parent node
        split_params = {}
        for node_id in parent_nodes:
            indices, labels = node_samples[node_id]
            
            if len(indices) <= 1 or len(np.unique(labels)) <= 1:
                continue  # Skip nodes with too few samples or pure class distribution
            
            # Get a random subset of features to consider
            feature_indices = self._get_feature_indices(n_features)
            
            # Find the best split
            best_feature_idx, best_threshold, _, best_is_cat, best_cats_left = optimize_split(
                X[indices], labels, feature_indices, self.n_classes,
                is_categorical=is_categorical, feature_bins=feature_bins
            )
            
            if best_feature_idx >= 0:
                split_params[node_id] = (best_feature_idx, best_threshold, best_is_cat, best_cats_left)
        
        # Build temporary nodes (2 per parent node if possible)
        temp_nodes = self._build_temporary_nodes(X, y, parent_nodes, node_samples)
        
        # Cluster the temporary nodes into n_child_nodes clusters
        node_to_cluster = self._cluster_nodes(temp_nodes, n_child_nodes)
        
        # Create branch assignments based on clustering
        left_child = {}
        right_child = {}
        
        for node_id in parent_nodes:
            if node_id not in split_params:
                continue
            
            # Find the temporary left and right children for this parent
            left_temp_id = None
            right_temp_id = None
            
            for temp_id, node_data in temp_nodes.items():
                if node_data['parent'] == node_id:
                    if node_data['is_left_child']:
                        left_temp_id = temp_id
                    else:
                        right_temp_id = temp_id
            
            # Assign left and right children based on clustering
            if left_temp_id is not None and left_temp_id in node_to_cluster:
                left_child[node_id] = node_to_cluster[left_temp_id]
            else:
                # Fallback in case left child wasn't created
                left_child[node_id] = 0 if n_child_nodes > 0 else None
            
            if right_temp_id is not None and right_temp_id in node_to_cluster:
                right_child[node_id] = node_to_cluster[right_temp_id]
            else:
                # Fallback in case right child wasn't created
                right_child[node_id] = 0 if n_child_nodes > 0 else None
        
        return split_params, left_child, right_child
