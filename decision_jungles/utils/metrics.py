"""
Metrics utilities for Decision Jungles.

This module provides functions for measuring memory usage, performance,
and other metrics for Decision Jungles.
"""

import numpy as np
import time
from typing import Any, Dict, Optional, Union, List, TypeVar, cast
from sklearn.ensemble import RandomForestClassifier

# Define a type variable for the DecisionJungleClassifier
DecisionJungleClassifier = TypeVar('DecisionJungleClassifier')
import sys


def memory_usage(obj: Any) -> int:
    """
    Calculate the memory usage of an object in bytes.
    
    Args:
        obj (Any): The object to measure.
        
    Returns:
        int: Memory usage in bytes.
    """
    if hasattr(obj, 'get_memory_usage'):
        return obj.get_memory_usage()
    else:
        return sys.getsizeof(obj)


def compare_memory_usage(jungle: DecisionJungleClassifier, 
                     forest: RandomForestClassifier) -> Dict[str, Union[int, float]]:
    """
    Compare memory usage between a Decision Jungle and a Random Forest.
    
    Args:
        jungle: A fitted DecisionJungleClassifier.
        forest: A fitted RandomForestClassifier.
        
    Returns:
        Dict[str, int]: Memory usage in bytes for both models.
    """
    jungle_memory = jungle.get_memory_usage()
    
    # For scikit-learn's RandomForestClassifier, we need to estimate
    forest_memory = sum(memory_usage(tree) for tree in forest.estimators_)
    
    return {
        'jungle_memory': jungle_memory,
        'forest_memory': forest_memory,
        'memory_ratio': forest_memory / jungle_memory if jungle_memory else float('inf')
    }


def measure_prediction_time(model: Any, X: np.ndarray, n_repeats: int = 10) -> Dict[str, float]:
    """
    Measure the prediction time for a model.
    
    Args:
        model: A fitted classifier with a predict method.
        X (np.ndarray): Input feature matrix.
        n_repeats (int): Number of times to repeat the prediction.
        
    Returns:
        Dict[str, float]: Prediction time statistics.
    """
    times = []
    
    for _ in range(n_repeats):
        start_time = time.time()
        model.predict(X)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }


def count_feature_evaluations(model: Any, X: np.ndarray) -> Dict[str, Union[float, str]]:
    """
    Count the number of feature evaluations required for prediction.
    
    Note: This is an approximation based on the maximum depth and number of DAGs.
    
    Args:
        model: A fitted DecisionJungleClassifier.
        X (np.ndarray): Input feature matrix.
        
    Returns:
        Dict[str, float]: Feature evaluation statistics.
    """
    if hasattr(model, 'get_max_depth') and hasattr(model, 'n_estimators'):
        max_depth = model.get_max_depth()
        n_estimators = model.n_estimators
        
        # Maximum number of feature evaluations per sample
        max_evaluations = max_depth * n_estimators
        
        # Total evaluations for all samples
        total_evaluations = max_evaluations * X.shape[0]
        
        return {
            'max_evaluations_per_sample': max_evaluations,
            'total_evaluations': total_evaluations,
            'evaluations_per_sample_avg': total_evaluations / X.shape[0]
        }
    else:
        # For other models, we can't easily compute this
        return {
            'max_evaluations_per_sample': 'unknown',
            'total_evaluations': 'unknown',
            'evaluations_per_sample_avg': 'unknown'
        }


def jaccard_index(y_true: np.ndarray, y_pred: np.ndarray, 
                 labels: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate the Jaccard index (intersection over union) for each class.
    
    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        labels (np.ndarray, optional): The set of labels to include.
        
    Returns:
        Dict[str, float]: Jaccard index for each class and average.
    """
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    jaccard_scores = {}
    
    for label in labels:
        # Create binary masks for this class
        true_mask = (y_true == label)
        pred_mask = (y_pred == label)
        
        # Calculate intersection and union
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        # Calculate Jaccard index
        if union > 0:
            jaccard_scores[f'class_{label}'] = intersection / union
        else:
            jaccard_scores[f'class_{label}'] = 1.0 if not np.any(true_mask) and not np.any(pred_mask) else 0.0
    
    # Calculate average Jaccard index
    jaccard_scores['average'] = np.mean(list(jaccard_scores.values()))
    
    return jaccard_scores
