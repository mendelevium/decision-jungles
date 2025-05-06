"""
Memory profiling utilities for Decision Jungles.

This module provides functions to measure and compare memory usage
of Decision Jungles and other models.
"""

import os
import gc
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import psutil

# Try to import optional memory profiling libraries
try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import pympler.asizeof as asizeof
    HAS_PYMPLER = True
except ImportError:
    HAS_PYMPLER = False


def get_process_memory() -> Tuple[float, float]:
    """
    Get current memory usage of the Python process.
    
    Returns:
        Tuple[float, float]: RSS (Resident Set Size) and VMS (Virtual Memory Size) in MB.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024), memory_info.vms / (1024 * 1024)


def measure_model_memory(model: Any, detailed: bool = False) -> Dict[str, float]:
    """
    Measure memory usage of a machine learning model.
    
    Args:
        model: The model to measure.
        detailed: Whether to provide detailed memory breakdown (requires pympler).
        
    Returns:
        Dict[str, float]: Dictionary with memory metrics in MB.
    """
    # Force garbage collection to get more accurate measurements
    gc.collect()
    
    result = {}
    
    # Get process memory before
    rss_before, vms_before = get_process_memory()
    
    # Different measurement techniques
    if hasattr(model, 'get_memory_usage'):
        # Our custom memory calculation for Decision Jungles
        result['reported_bytes'] = model.get_memory_usage()
        result['reported_mb'] = result['reported_bytes'] / (1024 * 1024)
    
    # Use pympler for detailed object size analysis if available
    if HAS_PYMPLER and detailed:
        result['pympler_bytes'] = asizeof.asizeof(model)
        result['pympler_mb'] = result['pympler_bytes'] / (1024 * 1024)
        
        # Get detailed breakdown for components
        if hasattr(model, 'dags_'):
            dags_size = asizeof.asizeof(model.dags_)
            result['dags_mb'] = dags_size / (1024 * 1024)
            
            # Individual DAG sizes
            dag_sizes = [asizeof.asizeof(dag) for dag in model.dags_]
            result['dag_sizes_mb'] = [size / (1024 * 1024) for size in dag_sizes]
            result['avg_dag_mb'] = np.mean(result['dag_sizes_mb'])
            
        if hasattr(model, 'estimators_'):
            estimators_size = asizeof.asizeof(model.estimators_)
            result['estimators_mb'] = estimators_size / (1024 * 1024)
    
    # Get process memory after
    rss_after, vms_after = get_process_memory()
    
    # Calculate change in memory
    result['rss_mb'] = rss_after
    result['vms_mb'] = vms_after
    result['rss_delta_mb'] = rss_after - rss_before
    result['vms_delta_mb'] = vms_after - vms_before
    
    return result


def memory_usage_vs_accuracy(
    jungle_constructor: Callable[..., Any],
    forest_constructor: Callable[..., Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    param_range: List[int],
    param_name: str = 'n_estimators',
    title: str = 'Memory Usage vs. Accuracy',
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compare memory usage vs. accuracy for Decision Jungles and Random Forests.
    
    Args:
        jungle_constructor: Function that takes a parameter value and returns a DecisionJungleClassifier.
        forest_constructor: Function that takes a parameter value and returns a RandomForestClassifier.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        param_range: List of parameter values to test.
        param_name: Name of the parameter being varied.
        title: Title for the plot.
        save_path: Path to save the plot image. If None, the plot is displayed.
        
    Returns:
        Dict[str, Dict[str, List[float]]]: Dictionary with memory usage and accuracy metrics.
    """
    # Initialize results
    results = {
        'jungle': {'memory_mb': [], 'accuracy': [], 'fit_time': []},
        'forest': {'memory_mb': [], 'accuracy': [], 'fit_time': []}
    }
    
    # Test each parameter value
    for param_value in param_range:
        # Train and measure Decision Jungle
        jungle = jungle_constructor(param_value)
        start_time = time.time()
        jungle.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        # Measure memory and accuracy
        memory_stats = measure_model_memory(jungle)
        accuracy = jungle.score(X_test, y_test)
        
        # Save results
        results['jungle']['memory_mb'].append(memory_stats.get('reported_mb', memory_stats['rss_delta_mb']))
        results['jungle']['accuracy'].append(accuracy)
        results['jungle']['fit_time'].append(fit_time)
        
        # Clean up to avoid memory accumulation
        del jungle
        gc.collect()
        
        # Train and measure Random Forest
        forest = forest_constructor(param_value)
        start_time = time.time()
        forest.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        # Measure memory and accuracy
        memory_stats = measure_model_memory(forest)
        accuracy = forest.score(X_test, y_test)
        
        # Save results
        results['forest']['memory_mb'].append(memory_stats['rss_delta_mb'])  # Use delta since it doesn't have get_memory_usage
        results['forest']['accuracy'].append(accuracy)
        results['forest']['fit_time'].append(fit_time)
        
        # Clean up to avoid memory accumulation
        del forest
        gc.collect()
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Memory vs. Parameter
    plt.subplot(1, 2, 1)
    plt.plot(param_range, results['jungle']['memory_mb'], 'o-', label='Decision Jungle')
    plt.plot(param_range, results['forest']['memory_mb'], 's-', label='Random Forest')
    plt.xlabel(param_name)
    plt.ylabel('Memory Usage (MB)')
    plt.title(f'Memory Usage vs. {param_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs. Memory
    plt.subplot(1, 2, 2)
    plt.plot(results['jungle']['memory_mb'], results['jungle']['accuracy'], 'o-', label='Decision Jungle')
    plt.plot(results['forest']['memory_mb'], results['forest']['accuracy'], 's-', label='Random Forest')
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Memory Usage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return results


def estimate_model_size(jungle_kwargs: Dict[str, Any], average_nodes_per_dag: Optional[int] = None) -> Dict[str, float]:
    """
    Estimate memory usage of a Decision Jungle without creating it.
    
    Args:
        jungle_kwargs: Dictionary of parameters for DecisionJungleClassifier.
        average_nodes_per_dag: Average number of nodes per DAG. If None, estimated based on parameters.
        
    Returns:
        Dict[str, float]: Estimated memory usage in bytes and MB.
    """
    # Extract parameters or use defaults
    n_estimators = jungle_kwargs.get('n_estimators', 10)
    max_width = jungle_kwargs.get('max_width', 256)
    max_depth = jungle_kwargs.get('max_depth', 10)
    n_classes = jungle_kwargs.get('n_classes', 2)
    
    # Estimate nodes per DAG if not provided
    if average_nodes_per_dag is None:
        if max_depth is None:
            max_depth = 10  # Default assumption
            
        # Rough estimate: for each level, we have at most max_width nodes
        # but this assumes full width at each level which is usually not the case
        # Let's assume 50% occupancy
        estimated_nodes = 0
        for d in range(max_depth + 1):
            level_width = min(max_width, 2**d)
            estimated_nodes += level_width * 0.5
            
        average_nodes_per_dag = int(estimated_nodes)
    
    # Memory per node (approximate):
    # - Split node: feature index (4 bytes) + threshold (8 bytes) + 
    #              pointers (2*4 bytes) + other attrs (~16 bytes) = ~36 bytes
    # - Leaf node: class distribution (n_classes * 8 bytes) + other attrs (~16 bytes)
    split_node_size = 36
    leaf_node_size = 16 + n_classes * 8
    
    # Assume half are split nodes and half are leaf nodes
    avg_node_size = (split_node_size + leaf_node_size) / 2
    
    # Total estimated memory
    total_bytes = n_estimators * average_nodes_per_dag * avg_node_size
    
    # Add overhead for the ensemble structure (~10%)
    total_bytes *= 1.1
    
    return {
        'estimated_bytes': total_bytes,
        'estimated_mb': total_bytes / (1024 * 1024),
        'estimated_nodes': n_estimators * average_nodes_per_dag
    }