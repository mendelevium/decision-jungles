"""
Objective functions for training Decision Jungles.

This module contains implementations of various objective functions used for
training Decision Jungles, primarily based on entropy and information gain.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set, Any, cast


def entropy(y: np.ndarray, n_classes: Optional[int] = None) -> float:
    """
    Calculate the Shannon entropy of a class distribution.
    
    Args:
        y (np.ndarray): Array of class labels.
        n_classes (int, optional): Number of classes. If None, determined from y.
        
    Returns:
        float: The entropy value.
    """
    if len(y) == 0:
        return 0.0
    
    if n_classes is None:
        n_classes = int(np.max(y) + 1)
    
    # Calculate class probabilities
    class_counts = np.bincount(y.astype(np.int32), minlength=n_classes)
    probs = class_counts / len(y)
    
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    
    # Calculate entropy: -sum(p_i * log2(p_i))
    return -np.sum(probs * np.log2(probs))


def weighted_entropy_sum(sets: List[np.ndarray], n_classes: Optional[int] = None) -> float:
    """
    Calculate the weighted sum of entropies across multiple sets.
    
    This is the objective function used for optimizing Decision Jungles.
    
    Args:
        sets (List[np.ndarray]): List of arrays containing class labels.
        n_classes (int, optional): Number of classes. If None, determined from data.
        
    Returns:
        float: The weighted sum of entropies.
    """
    if not sets:
        return 0.0
    
    # Determine number of classes if not provided
    if n_classes is None:
        all_labels = np.concatenate(sets)
        if len(all_labels) > 0:
            n_classes = int(np.max(all_labels) + 1)
        else:
            return 0.0
    
    # Calculate the weighted sum of entropies
    total_samples = sum(len(s) for s in sets)
    if total_samples == 0:
        return 0.0
    
    weighted_sum = 0.0
    for s in sets:
        if len(s) > 0:
            weight = len(s) / total_samples
            weighted_sum += weight * entropy(s, n_classes)
    
    return weighted_sum


def information_gain(y_parent: np.ndarray, y_children: List[np.ndarray], 
                     n_classes: Optional[int] = None) -> float:
    """
    Calculate the information gain from a split.
    
    Args:
        y_parent (np.ndarray): Class labels at the parent node.
        y_children (List[np.ndarray]): Class labels at child nodes.
        n_classes (int, optional): Number of classes. If None, determined from data.
        
    Returns:
        float: The information gain.
    """
    if len(y_parent) == 0:
        return 0.0
    
    # Determine number of classes if not provided
    if n_classes is None:
        n_classes = int(np.max(y_parent) + 1)
    
    # Calculate parent entropy
    parent_entropy = entropy(y_parent, n_classes)
    
    # Calculate weighted child entropy
    child_entropy = weighted_entropy_sum(y_children, n_classes)
    
    # Information gain is the difference
    return parent_entropy - child_entropy


def optimize_split(X: np.ndarray, y: np.ndarray, feature_indices: Optional[List[int]] = None,
                  n_classes: Optional[int] = None, n_thresholds: int = 10,
                  is_categorical: Optional[np.ndarray] = None,
                  feature_bins: Optional[Dict[int, Dict[Any, int]]] = None) -> Tuple[int, float, float, bool, Optional[Set[Any]]]:
    """
    Find the optimal split parameters for a node.
    
    Args:
        X (np.ndarray): Feature matrix of samples.
        y (np.ndarray): Class labels of samples.
        feature_indices (List[int], optional): Indices of features to consider.
            If None, all features are considered.
        n_classes (int, optional): Number of classes. If None, determined from y.
        n_thresholds (int): Number of threshold values to evaluate per feature.
        is_categorical (np.ndarray, optional): Boolean mask indicating which features are categorical.
        feature_bins (Dict, optional): Dictionary mapping feature indices to category-to-bin mappings.
        
    Returns:
        Tuple[int, float, float, bool, Optional[Set]]: The optimal feature index, threshold value,
            corresponding information gain, whether the feature is categorical, and
            for categorical features, the set of categories that go left.
    """
    if len(y) <= 1:
        return (-1, 0.0, 0.0, False, None)  # Not enough samples to split
    
    if n_classes is None:
        n_classes = int(np.max(y) + 1)
    
    if feature_indices is None:
        feature_indices = list(range(X.shape[1]))
    
    best_info_gain = -1.0
    best_feature = -1
    best_threshold = 0.0
    best_is_categorical = False
    best_categories_left = None
    
    for feature in feature_indices:
        # Check if this is a categorical feature
        is_cat = is_categorical is not None and feature < len(is_categorical) and is_categorical[feature]
        
        if is_cat and feature_bins is not None and feature in feature_bins:
            # Handle categorical features
            categories = list(feature_bins[feature].keys())
            
            # Skip if there's only one category
            if len(categories) <= 1:
                continue
                
            # Try different category combinations for splitting
            # For simplicity, we'll use a greedy approach: 
            # Sort categories by their proportion of positive class and try different cutoffs
            
            # First, calculate the proportion of positive class for each category
            category_stats = {}
            for cat in categories:
                # Find samples with this category
                cat_mask = np.isclose(X[:, feature], cat)
                if not np.any(cat_mask):
                    continue
                    
                cat_y = y[cat_mask]
                if len(cat_y) == 0:
                    continue
                    
                # Calculate class distribution for this category
                class_counts = np.bincount(cat_y.astype(np.int32), minlength=n_classes)
                category_stats[cat] = class_counts / np.sum(class_counts)
            
            # Sort categories by the proportion of the first class (or other criterion)
            sorted_cats = sorted(category_stats.keys(), 
                                key=lambda c: category_stats[c][0] if len(category_stats[c]) > 0 else 0)
            
            # Try different partitioning points
            for i in range(1, len(sorted_cats)):
                # Categories going left
                left_cats = set(sorted_cats[:i])
                
                # Create masks for the split
                left_mask = np.zeros(X.shape[0], dtype=bool)
                for j in range(X.shape[0]):
                    if not np.isnan(X[j, feature]) and X[j, feature] in left_cats:
                        left_mask[j] = True
                
                right_mask = ~left_mask
                
                # Skip if the split doesn't separate samples
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate information gain
                ig = information_gain(y, [y_left, y_right], n_classes)
                
                if ig > best_info_gain:
                    best_info_gain = ig
                    best_feature = feature
                    best_threshold = 0.0  # Not used for categorical features
                    best_is_categorical = True
                    best_categories_left = left_cats
                    
        else:
            # Handle numerical features with the original approach
            # Get unique values for this feature, sorted
            unique_values = np.unique(X[:, feature])
            non_nan_values = unique_values[~np.isnan(unique_values)]
            values = np.sort(non_nan_values)
            
            if len(values) <= 1:
                continue  # Skip features with only one unique value
            
            # Generate thresholds between unique values
            thresholds = []
            for i in range(len(values) - 1):
                thresholds.append((values[i] + values[i + 1]) / 2)
            
            # If too many thresholds, sample a subset
            if len(thresholds) > n_thresholds:
                indices = np.linspace(0, len(thresholds) - 1, n_thresholds, dtype=int)
                thresholds = [thresholds[i] for i in indices]
            
            # Evaluate each threshold
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Handle NaN values - send them to the right
                nan_mask = np.isnan(X[:, feature])
                left_mask[nan_mask] = False
                right_mask[nan_mask] = True
                
                # Skip if the split doesn't separate samples
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate information gain
                ig = information_gain(y, [y_left, y_right], n_classes)
                
                if ig > best_info_gain:
                    best_info_gain = ig
                    best_feature = feature
                    best_threshold = threshold
                    best_is_categorical = False
                    best_categories_left = None
    
    return (best_feature, best_threshold, best_info_gain, best_is_categorical, best_categories_left)
