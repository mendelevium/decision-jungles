"""
Regression objective functions for training Decision Jungles.

This module contains implementations of objective functions used for
regression tasks in Decision Jungles, primarily based on variance reduction.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set, Any, cast


def mse(y: np.ndarray) -> float:
    """
    Calculate the mean squared error (variance) of a set of target values.
    
    Args:
        y (np.ndarray): Array of target values.
        
    Returns:
        float: The mean squared error.
    """
    if len(y) <= 1:
        return 0.0
    
    # MSE is the variance of the target values
    return np.var(y)


def mae(y: np.ndarray) -> float:
    """
    Calculate the mean absolute error of a set of target values.
    
    This is defined as the mean absolute deviation from the median.
    
    Args:
        y (np.ndarray): Array of target values.
        
    Returns:
        float: The mean absolute error.
    """
    if len(y) <= 1:
        return 0.0
    
    median = np.median(y)
    return np.mean(np.abs(y - median))


def weighted_error_sum(sets: List[np.ndarray], criterion: str = "mse") -> float:
    """
    Calculate the weighted sum of errors across multiple sets.
    
    This is the objective function used for optimizing Decision Jungles regression.
    
    Args:
        sets (List[np.ndarray]): List of arrays containing target values.
        criterion (str): The criterion to use ("mse" or "mae").
        
    Returns:
        float: The weighted sum of errors.
    """
    if not sets:
        return 0.0
    
    # Calculate the weighted sum of errors
    total_samples = sum(len(s) for s in sets)
    if total_samples == 0:
        return 0.0
    
    weighted_sum = 0.0
    for s in sets:
        if len(s) > 0:
            weight = len(s) / total_samples
            if criterion == "mae":
                weighted_sum += weight * mae(s)
            else:  # default to mse
                weighted_sum += weight * mse(s)
    
    return weighted_sum


def impurity_reduction(y_parent: np.ndarray, y_children: List[np.ndarray], criterion: str = "mse") -> float:
    """
    Calculate the impurity reduction (variance reduction) from a split.
    
    Args:
        y_parent (np.ndarray): Target values at the parent node.
        y_children (List[np.ndarray]): Target values at child nodes.
        criterion (str): The criterion to use ("mse" or "mae").
        
    Returns:
        float: The impurity reduction.
    """
    if len(y_parent) == 0:
        return 0.0
    
    # Calculate parent impurity
    if criterion == "mae":
        parent_impurity = mae(y_parent)
    else:  # default to mse
        parent_impurity = mse(y_parent)
    
    # Calculate weighted child impurity
    child_impurity = weighted_error_sum(y_children, criterion)
    
    # Impurity reduction is the difference
    return parent_impurity - child_impurity


def optimize_split_regression(X: np.ndarray, y: np.ndarray, feature_indices: Optional[List[int]] = None,
                             n_thresholds: int = 10, criterion: str = "mse",
                             is_categorical: Optional[np.ndarray] = None,
                             feature_bins: Optional[Dict[int, Dict[Any, int]]] = None) -> Tuple[int, float, float, bool, Optional[Set[Any]]]:
    """
    Find the optimal split parameters for a regression node.
    
    Args:
        X (np.ndarray): Feature matrix of samples.
        y (np.ndarray): Target values of samples.
        feature_indices (List[int], optional): Indices of features to consider.
            If None, all features are considered.
        n_thresholds (int): Number of threshold values to evaluate per feature.
        criterion (str): The criterion to use ("mse" or "mae").
        is_categorical (np.ndarray, optional): Boolean mask indicating which features are categorical.
        feature_bins (Dict, optional): Dictionary mapping feature indices to category-to-bin mappings.
        
    Returns:
        Tuple[int, float, float, bool, Optional[Set]]: The optimal feature index, threshold value,
            corresponding impurity reduction, whether the feature is categorical, and
            for categorical features, the set of categories that go left.
    """
    if len(y) <= 1:
        return (-1, 0.0, 0.0, False, None)  # Not enough samples to split
    
    if feature_indices is None:
        feature_indices = list(range(X.shape[1]))
    
    best_impurity_reduction = -1.0
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
                
            # For regression, we'll group categories by their mean target value
            # Sort categories by their mean target value
            category_stats = {}
            for cat in categories:
                # Find samples with this category
                cat_mask = np.isclose(X[:, feature], cat)
                if not np.any(cat_mask):
                    continue
                    
                cat_y = y[cat_mask]
                if len(cat_y) == 0:
                    continue
                    
                # Calculate mean target value for this category
                category_stats[cat] = np.mean(cat_y)
            
            # Sort categories by mean target value
            sorted_cats = sorted(category_stats.keys(), key=lambda c: category_stats[c])
            
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
                
                # Calculate impurity reduction
                ir = impurity_reduction(y, [y_left, y_right], criterion)
                
                if ir > best_impurity_reduction:
                    best_impurity_reduction = ir
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
                
                # Calculate impurity reduction
                ir = impurity_reduction(y, [y_left, y_right], criterion)
                
                if ir > best_impurity_reduction:
                    best_impurity_reduction = ir
                    best_feature = feature
                    best_threshold = threshold
                    best_is_categorical = False
                    best_categories_left = None
    
    return (best_feature, best_threshold, best_impurity_reduction, best_is_categorical, best_categories_left)