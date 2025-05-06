"""
Cythonized version of the LSearch optimization algorithm for Decision Jungles.

This module provides a Cython implementation of the core LSearch algorithm
to optimize performance for training Decision Jungles.
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport log
import cython

# Define types
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t
ctypedef np.uint8_t BOOL_t

# Shannon entropy calculation
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double entropy(np.ndarray[DTYPE_t, ndim=1] class_distribution) nogil:
    """
    Calculate the Shannon entropy of a class distribution.
    
    Args:
        class_distribution: Array of class counts/weights
        
    Returns:
        Shannon entropy value
    """
    cdef double total = 0.0
    cdef double p = 0.0
    cdef double result = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t n = class_distribution.shape[0]
    
    for i in range(n):
        total += class_distribution[i]
    
    if total <= 0.0:
        return 0.0
    
    for i in range(n):
        if class_distribution[i] > 0.0:
            p = class_distribution[i] / total
            result -= p * log(p)
    
    return result

# Calculate weighted entropy of child distributions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double weighted_entropy(
        np.ndarray[DTYPE_t, ndim=1] left_dist,
        np.ndarray[DTYPE_t, ndim=1] right_dist) nogil:
    """
    Calculate the weighted entropy of two class distributions.
    
    Args:
        left_dist: Array of class counts/weights for left branch
        right_dist: Array of class counts/weights for right branch
        
    Returns:
        Weighted entropy value
    """
    cdef double left_total = 0.0
    cdef double right_total = 0.0
    cdef double total = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t n = left_dist.shape[0]
    
    for i in range(n):
        left_total += left_dist[i]
        right_total += right_dist[i]
    
    total = left_total + right_total
    
    if total <= 0.0:
        return 0.0
    
    return (left_total / total) * entropy(left_dist) + (right_total / total) * entropy(right_dist)

# Find the best split for a feature
@cython.boundscheck(False)
@cython.wraparound(False)
def find_best_split(
        np.ndarray[DTYPE_t, ndim=2] X,
        np.ndarray[ITYPE_t, ndim=1] y,
        np.ndarray[ITYPE_t, ndim=1] sample_indices,
        int feature_idx,
        int n_classes):
    """
    Find the best split point for a given feature.
    
    Args:
        X: Feature matrix
        y: Class labels
        sample_indices: Indices of samples to consider
        feature_idx: Index of the feature to split on
        n_classes: Number of classes
        
    Returns:
        Tuple of (threshold, information gain, left_indices, right_indices)
    """
    cdef Py_ssize_t n_samples = sample_indices.shape[0]
    cdef Py_ssize_t i, j
    cdef double best_gain = -1.0
    cdef double best_threshold = 0.0
    cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(n_samples, dtype=np.float64)
    cdef np.ndarray[ITYPE_t, ndim=1] sorted_indices = np.zeros(n_samples, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] parent_dist = np.zeros(n_classes, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] left_dist = np.zeros(n_classes, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] right_dist = np.zeros(n_classes, dtype=np.float64)
    cdef double parent_entropy, current_gain, threshold
    
    # Get feature values for the samples
    for i in range(n_samples):
        values[i] = X[sample_indices[i], feature_idx]
        parent_dist[y[sample_indices[i]]] += 1.0
    
    # Calculate parent entropy
    parent_entropy = entropy(parent_dist)
    
    # Skip if parent entropy is 0 (all samples have the same class)
    if parent_entropy <= 0.0:
        return 0.0, 0.0, np.array([], dtype=np.int32), sample_indices
    
    # Sort values and corresponding sample indices
    sorted_indices = np.argsort(values)
    
    # Initialize distributions
    for i in range(n_classes):
        right_dist[i] = parent_dist[i]
        left_dist[i] = 0.0
    
    # Try different split points
    for i in range(1, n_samples):
        # Skip if the value is the same as the previous one
        if values[sorted_indices[i]] <= values[sorted_indices[i-1]]:
            continue
        
        # Move a sample from right to left
        sample_idx = sample_indices[sorted_indices[i-1]]
        class_idx = y[sample_idx]
        left_dist[class_idx] += 1.0
        right_dist[class_idx] -= 1.0
        
        # Calculate weighted entropy and information gain
        threshold = (values[sorted_indices[i]] + values[sorted_indices[i-1]]) / 2.0
        current_gain = parent_entropy - weighted_entropy(left_dist, right_dist)
        
        # Update best split
        if current_gain > best_gain:
            best_gain = current_gain
            best_threshold = threshold
    
    # If no good split found, return all samples to the right
    if best_gain <= 0.0:
        return 0.0, 0.0, np.array([], dtype=np.int32), sample_indices
    
    # Split samples based on the best threshold
    cdef np.ndarray[ITYPE_t, ndim=1] left_indices = np.zeros(n_samples, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] right_indices = np.zeros(n_samples, dtype=np.int32)
    cdef Py_ssize_t left_count = 0
    cdef Py_ssize_t right_count = 0
    
    for i in range(n_samples):
        if X[sample_indices[i], feature_idx] <= best_threshold:
            left_indices[left_count] = sample_indices[i]
            left_count += 1
        else:
            right_indices[right_count] = sample_indices[i]
            right_count += 1
    
    return best_threshold, best_gain, left_indices[:left_count], right_indices[:right_count]

# Objective function for branch assignment optimization
@cython.boundscheck(False)
@cython.wraparound(False)
def branch_assignment_objective(
        np.ndarray[BOOL_t, ndim=1] branch_assignments,
        np.ndarray[DTYPE_t, ndim=2] node_class_distributions):
    """
    Calculate the objective function value for branch assignments.
    
    Args:
        branch_assignments: Array of 0s and 1s for left and right branch assignments
        node_class_distributions: Class distributions for each parent node
        
    Returns:
        Objective function value (to be minimized)
    """
    cdef Py_ssize_t n_nodes = node_class_distributions.shape[0]
    cdef Py_ssize_t n_classes = node_class_distributions.shape[1]
    cdef Py_ssize_t i, j
    
    # Initialize left and right distributions
    cdef np.ndarray[DTYPE_t, ndim=1] left_dist = np.zeros(n_classes, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] right_dist = np.zeros(n_classes, dtype=np.float64)
    
    # Aggregate class distributions based on branch assignments
    for i in range(n_nodes):
        if branch_assignments[i] == 0:  # Left branch
            for j in range(n_classes):
                left_dist[j] += node_class_distributions[i, j]
        else:  # Right branch
            for j in range(n_classes):
                right_dist[j] += node_class_distributions[i, j]
    
    # Return weighted entropy (objective to minimize)
    return weighted_entropy(left_dist, right_dist)

# Optimize branch assignments
def optimize_branch_assignments(
        np.ndarray[DTYPE_t, ndim=2] node_class_distributions,
        int max_iterations=100):
    """
    Optimize branch assignments using a greedy algorithm.
    
    Args:
        node_class_distributions: Class distributions for each parent node
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (optimized branch assignments, objective value)
    """
    cdef Py_ssize_t n_nodes = node_class_distributions.shape[0]
    cdef Py_ssize_t i, iter_count
    cdef double current_obj, new_obj
    cdef np.ndarray[BOOL_t, ndim=1] best_assignments = np.zeros(n_nodes, dtype=np.uint8)
    cdef np.ndarray[BOOL_t, ndim=1] test_assignments = np.zeros(n_nodes, dtype=np.uint8)
    cdef bint improved
    
    # Initialize with all nodes assigned to the left branch
    current_obj = branch_assignment_objective(best_assignments, node_class_distributions)
    
    # Iteratively optimize assignments
    for iter_count in range(max_iterations):
        improved = False
        
        # Try flipping each node's assignment
        for i in range(n_nodes):
            # Copy current assignments
            test_assignments = best_assignments.copy()
            
            # Flip the assignment for node i
            test_assignments[i] = 1 - test_assignments[i]
            
            # Calculate new objective
            new_obj = branch_assignment_objective(test_assignments, node_class_distributions)
            
            # Update if improved
            if new_obj < current_obj:
                best_assignments = test_assignments.copy()
                current_obj = new_obj
                improved = True
        
        # Stop if no improvement
        if not improved:
            break
    
    return best_assignments, current_obj

# Main LSearch algorithm (Python wrapper for integration)
class CythLSearch:
    """
    Cythonized implementation of the LSearch algorithm.
    
    This class provides a wrapper for the Cythonized components of the
    LSearch algorithm to integrate with the Decision Jungle implementation.
    """
    
    def __init__(self, n_classes):
        """
        Initialize the LSearch algorithm.
        
        Args:
            n_classes: Number of classes in the classification problem
        """
        self.n_classes = n_classes
    
    def find_best_split(self, X, y, sample_indices, feature_idx):
        """
        Find the best split for a feature.
        
        Args:
            X: Feature matrix
            y: Class labels
            sample_indices: Indices of samples to consider
            feature_idx: Feature index to split on
            
        Returns:
            Tuple of (threshold, information gain, left_indices, right_indices)
        """
        return find_best_split(X, y, sample_indices, feature_idx, self.n_classes)
    
    def optimize_branch_assignments(self, node_class_distributions, max_iterations=100):
        """
        Optimize branch assignments.
        
        Args:
            node_class_distributions: Class distributions for each parent node
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (optimized branch assignments, objective value)
        """
        return optimize_branch_assignments(node_class_distributions, max_iterations)