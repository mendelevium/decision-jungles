"""
Benchmarking utilities for Decision Jungles.

This module provides functions to measure and compare performance
of different implementations and configurations of Decision Jungles.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from sklearn.datasets import make_classification, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split


def time_execution(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    """
    Measure execution time of a function.
    
    Args:
        func: Function to time.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        Tuple[float, Any]: Execution time in seconds and function result.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return execution_time, result


def benchmark_optimization(
    X: np.ndarray,
    y: np.ndarray,
    jungle_class: Any,
    n_repeats: int = 3,
    n_estimators: int = 5,
    max_width: int = 64,
    max_depth: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Benchmark standard vs optimized implementations.
    
    Args:
        X: Feature matrix.
        y: Target labels.
        jungle_class: Decision Jungle class to benchmark.
        n_repeats: Number of times to repeat each test for robustness.
        n_estimators: Number of estimators to use.
        max_width: Maximum width parameter.
        max_depth: Maximum depth parameter.
        random_state: Random state for reproducibility.
        
    Returns:
        pd.DataFrame: Benchmark results.
    """
    results = []
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Test configurations
    configs = [
        {"name": "Standard Implementation", "use_optimized": False},
        {"name": "Optimized Implementation", "use_optimized": True},
    ]
    
    for config in configs:
        for repeat in range(n_repeats):
            # Create model with this configuration
            model = jungle_class(
                n_estimators=n_estimators,
                max_width=max_width,
                max_depth=max_depth,
                random_state=random_state + repeat,
                use_optimized=config["use_optimized"],
            )
            
            # Time the training
            train_time, _ = time_execution(model.fit, X_train, y_train)
            
            # Time the prediction
            predict_time, y_pred = time_execution(model.predict, X_test)
            
            # Time probability prediction
            predict_proba_time, y_proba = time_execution(model.predict_proba, X_test)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_test)
            
            # Memory usage
            if hasattr(model, 'get_memory_usage'):
                memory_bytes = model.get_memory_usage()
                memory_mb = memory_bytes / (1024 * 1024)
            else:
                memory_bytes = None
                memory_mb = None
            
            # Node count
            if hasattr(model, 'get_node_count'):
                node_count = model.get_node_count()
            else:
                node_count = None
            
            # Store results
            results.append({
                "implementation": config["name"],
                "repeat": repeat + 1,
                "train_time": train_time,
                "predict_time": predict_time,
                "predict_proba_time": predict_proba_time,
                "accuracy": accuracy,
                "memory_bytes": memory_bytes,
                "memory_mb": memory_mb,
                "node_count": node_count,
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = results_df.groupby("implementation").agg({
        "train_time": ["mean", "std"],
        "predict_time": ["mean", "std"],
        "predict_proba_time": ["mean", "std"],
        "accuracy": ["mean", "std"],
        "memory_mb": ["mean"],
        "node_count": ["mean"],
    })
    
    # Calculate speedup
    if len(configs) > 1:
        baseline_time = summary.loc["Standard Implementation", ("train_time", "mean")]
        optimized_time = summary.loc["Optimized Implementation", ("train_time", "mean")]
        speedup = baseline_time / optimized_time
        print(f"Training time speedup: {speedup:.2f}x")
    
    return results_df


def benchmark_scaling(
    jungle_class: Any,
    sizes: List[int] = [1000, 10000, 50000],
    features: List[int] = [10, 50, 100],
    n_estimators: int = 5,
    max_width: int = 64,
    max_depth: int = 8,
    use_optimized: bool = True,
    random_state: int = 42,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Benchmark scaling behavior with dataset size and dimensionality.
    
    Args:
        jungle_class: Decision Jungle class to benchmark.
        sizes: List of dataset sizes to test.
        features: List of feature counts to test.
        n_estimators: Number of estimators to use.
        max_width: Maximum width parameter.
        max_depth: Maximum depth parameter.
        use_optimized: Whether to use optimized implementation.
        random_state: Random state for reproducibility.
        plot: Whether to generate plots.
        save_path: Path to save plot if generated.
        
    Returns:
        pd.DataFrame: Benchmark results.
    """
    results = []
    
    for n_samples in sizes:
        for n_features in features:
            # Generate synthetic data
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(5, n_features // 2),
                n_redundant=max(3, n_features // 5),
                n_classes=3,
                random_state=random_state,
            )
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
            
            # Create model
            model = jungle_class(
                n_estimators=n_estimators,
                max_width=max_width,
                max_depth=max_depth,
                random_state=random_state,
                use_optimized=use_optimized,
            )
            
            # Time the training
            print(f"Training with {n_samples} samples, {n_features} features...")
            train_time, _ = time_execution(model.fit, X_train, y_train)
            
            # Time the prediction
            predict_time, y_pred = time_execution(model.predict, X_test)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_test)
            
            # Store results
            results.append({
                "n_samples": n_samples,
                "n_features": n_features,
                "train_time": train_time,
                "predict_time": predict_time,
                "accuracy": accuracy,
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot scaling behavior
    if plot:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Training time vs dataset size
        plt.subplot(2, 2, 1)
        for n_features in features:
            subset = results_df[results_df['n_features'] == n_features]
            plt.plot(subset['n_samples'], subset['train_time'], 'o-', 
                     label=f'{n_features} features')
        plt.xlabel('Number of Samples')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Training time vs number of features
        plt.subplot(2, 2, 2)
        for n_samples in sizes:
            subset = results_df[results_df['n_samples'] == n_samples]
            plt.plot(subset['n_features'], subset['train_time'], 's-', 
                     label=f'{n_samples} samples')
        plt.xlabel('Number of Features')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time vs Feature Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Prediction time vs dataset size
        plt.subplot(2, 2, 3)
        for n_features in features:
            subset = results_df[results_df['n_features'] == n_features]
            plt.plot(subset['n_samples'], subset['predict_time'], 'o-', 
                     label=f'{n_features} features')
        plt.xlabel('Number of Samples')
        plt.ylabel('Prediction Time (s)')
        plt.title('Prediction Time vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy vs dataset size
        plt.subplot(2, 2, 4)
        for n_features in features:
            subset = results_df[results_df['n_features'] == n_features]
            plt.plot(subset['n_samples'], subset['accuracy'], 'o-', 
                     label=f'{n_features} features')
        plt.xlabel('Number of Samples')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    return results_df


def benchmark_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    jungle_class: Any,
    n_estimators_range: List[int] = [1, 5, 10, 20],
    max_width_range: List[int] = [16, 64, 256],
    max_depth_range: List[int] = [5, 10, None],
    use_optimized: bool = True,
    random_state: int = 42,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Benchmark performance across different hyperparameter settings.
    
    Args:
        X: Feature matrix.
        y: Target labels.
        jungle_class: Decision Jungle class to benchmark.
        n_estimators_range: List of n_estimators values to test.
        max_width_range: List of max_width values to test.
        max_depth_range: List of max_depth values to test.
        use_optimized: Whether to use optimized implementation.
        random_state: Random state for reproducibility.
        plot: Whether to generate plots.
        save_path: Path to save plot if generated.
        
    Returns:
        pd.DataFrame: Benchmark results.
    """
    results = []
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Default values
    default_n_estimators = 10
    default_max_width = 64
    default_max_depth = 8
    
    # Test n_estimators
    for n_estimators in n_estimators_range:
        model = jungle_class(
            n_estimators=n_estimators,
            max_width=default_max_width,
            max_depth=default_max_depth,
            random_state=random_state,
            use_optimized=use_optimized,
        )
        
        # Time the training
        train_time, _ = time_execution(model.fit, X_train, y_train)
        
        # Time the prediction
        predict_time, y_pred = time_execution(model.predict, X_test)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        
        # Memory usage
        if hasattr(model, 'get_memory_usage'):
            memory_bytes = model.get_memory_usage()
            memory_mb = memory_bytes / (1024 * 1024)
        else:
            memory_bytes = None
            memory_mb = None
        
        # Node count
        if hasattr(model, 'get_node_count'):
            node_count = model.get_node_count()
        else:
            node_count = None
        
        # Store results
        results.append({
            "param_type": "n_estimators",
            "param_value": n_estimators,
            "train_time": train_time,
            "predict_time": predict_time,
            "accuracy": accuracy,
            "memory_mb": memory_mb,
            "node_count": node_count,
        })
    
    # Test max_width
    for max_width in max_width_range:
        model = jungle_class(
            n_estimators=default_n_estimators,
            max_width=max_width,
            max_depth=default_max_depth,
            random_state=random_state,
            use_optimized=use_optimized,
        )
        
        # Time the training
        train_time, _ = time_execution(model.fit, X_train, y_train)
        
        # Time the prediction
        predict_time, y_pred = time_execution(model.predict, X_test)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        
        # Memory usage
        if hasattr(model, 'get_memory_usage'):
            memory_bytes = model.get_memory_usage()
            memory_mb = memory_bytes / (1024 * 1024)
        else:
            memory_bytes = None
            memory_mb = None
        
        # Node count
        if hasattr(model, 'get_node_count'):
            node_count = model.get_node_count()
        else:
            node_count = None
        
        # Store results
        results.append({
            "param_type": "max_width",
            "param_value": max_width,
            "train_time": train_time,
            "predict_time": predict_time,
            "accuracy": accuracy,
            "memory_mb": memory_mb,
            "node_count": node_count,
        })
    
    # Test max_depth
    for max_depth in max_depth_range:
        # Convert None to 'None' for display
        display_value = 'None' if max_depth is None else max_depth
        
        model = jungle_class(
            n_estimators=default_n_estimators,
            max_width=default_max_width,
            max_depth=max_depth,
            random_state=random_state,
            use_optimized=use_optimized,
        )
        
        # Time the training
        train_time, _ = time_execution(model.fit, X_train, y_train)
        
        # Time the prediction
        predict_time, y_pred = time_execution(model.predict, X_test)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        
        # Memory usage
        if hasattr(model, 'get_memory_usage'):
            memory_bytes = model.get_memory_usage()
            memory_mb = memory_bytes / (1024 * 1024)
        else:
            memory_bytes = None
            memory_mb = None
        
        # Node count
        if hasattr(model, 'get_node_count'):
            node_count = model.get_node_count()
        else:
            node_count = None
        
        # Store results
        results.append({
            "param_type": "max_depth",
            "param_value": display_value,  # Use string for display
            "param_value_numeric": -1 if max_depth is None else max_depth,  # For sorting
            "train_time": train_time,
            "predict_time": predict_time,
            "accuracy": accuracy,
            "memory_mb": memory_mb,
            "node_count": node_count,
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    if plot:
        plt.figure(figsize=(15, 12))
        
        # Plot n_estimators results
        n_est_results = results_df[results_df['param_type'] == 'n_estimators']
        
        plt.subplot(3, 3, 1)
        plt.plot(n_est_results['param_value'], n_est_results['train_time'], 'o-')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time vs. n_estimators')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 2)
        plt.plot(n_est_results['param_value'], n_est_results['accuracy'], 'o-')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. n_estimators')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 3)
        plt.plot(n_est_results['param_value'], n_est_results['memory_mb'], 'o-')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs. n_estimators')
        plt.grid(True, alpha=0.3)
        
        # Plot max_width results
        width_results = results_df[results_df['param_type'] == 'max_width']
        
        plt.subplot(3, 3, 4)
        plt.plot(width_results['param_value'], width_results['train_time'], 'o-')
        plt.xlabel('Max Width')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time vs. max_width')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 5)
        plt.plot(width_results['param_value'], width_results['accuracy'], 'o-')
        plt.xlabel('Max Width')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. max_width')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 6)
        plt.plot(width_results['param_value'], width_results['memory_mb'], 'o-')
        plt.xlabel('Max Width')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs. max_width')
        plt.grid(True, alpha=0.3)
        
        # Plot max_depth results
        depth_results = results_df[results_df['param_type'] == 'max_depth']
        # For display purposes, sort by the numeric value
        if 'param_value_numeric' in depth_results.columns:
            depth_results = depth_results.sort_values('param_value_numeric')
        
        plt.subplot(3, 3, 7)
        plt.plot(depth_results['param_value'], depth_results['train_time'], 'o-')
        plt.xlabel('Max Depth')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time vs. max_depth')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 8)
        plt.plot(depth_results['param_value'], depth_results['accuracy'], 'o-')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. max_depth')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 9)
        plt.plot(depth_results['param_value'], depth_results['memory_mb'], 'o-')
        plt.xlabel('Max Depth')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs. max_depth')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    return results_df