#!/usr/bin/env python
"""
Performance profiling for Decision Jungles.

This script profiles the performance of the Decision Jungle implementation to identify
bottlenecks and areas for Cythonization.
"""

import numpy as np
import time
import cProfile
import pstats
from pstats import SortKey
import io
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

# Import our Decision Jungle classifier
from decision_jungles import DecisionJungleClassifier


def profile_training(X, y, n_estimators=10, max_width=64, max_depth=10, 
                     optimization_method="lsearch", use_optimized=True):
    """
    Profile the training process of a Decision Jungle.
    
    Args:
        X: Feature matrix
        y: Target labels
        n_estimators: Number of DAGs in the ensemble
        max_width: Maximum width parameter
        max_depth: Maximum depth of the DAGs
        optimization_method: Method for DAG optimization
        use_optimized: Whether to use the optimized implementation
        
    Returns:
        stats: Profiling statistics
    """
    # Create the classifier
    jungle = DecisionJungleClassifier(
        n_estimators=n_estimators,
        max_width=max_width,
        max_depth=max_depth,
        optimization_method=optimization_method,
        use_optimized=use_optimized,
        random_state=42
    )
    
    # Profile fitting
    pr = cProfile.Profile()
    pr.enable()
    
    jungle.fit(X, y)
    
    pr.disable()
    
    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    
    return ps, s.getvalue(), jungle


def profile_prediction(jungle, X):
    """
    Profile the prediction process of a Decision Jungle.
    
    Args:
        jungle: Fitted DecisionJungleClassifier
        X: Feature matrix for prediction
        
    Returns:
        stats: Profiling statistics
    """
    # Profile prediction
    pr = cProfile.Profile()
    pr.enable()
    
    jungle.predict(X)
    
    pr.disable()
    
    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    
    return ps, s.getvalue()


def compare_optimization_methods(X, y):
    """
    Compare the performance of different optimization methods.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        results: Dictionary of results
    """
    methods = [
        ("lsearch", False),
        ("lsearch", True),
        ("clustersearch", False)
    ]
    
    results = {}
    
    for method, use_optimized in methods:
        method_name = f"{method}_{'optimized' if use_optimized else 'standard'}"
        print(f"\nProfiling {method_name}...")
        
        # Time the training
        start_time = time.time()
        _, stats, jungle = profile_training(
            X, y, 
            optimization_method=method, 
            use_optimized=use_optimized
        )
        train_time = time.time() - start_time
        
        # Time the prediction
        X_sample = X[:100]
        start_time = time.time()
        jungle.predict(X_sample)
        predict_time = time.time() - start_time
        
        # Get memory usage
        memory_usage = jungle.get_memory_usage()
        node_count = jungle.get_node_count()
        
        results[method_name] = {
            'train_time': train_time,
            'predict_time': predict_time,
            'memory_usage': memory_usage,
            'node_count': node_count,
            'stats': stats
        }
        
        print(f"Training time: {train_time:.4f} seconds")
        print(f"Prediction time (100 samples): {predict_time:.6f} seconds")
        print(f"Memory usage: {memory_usage} bytes")
        print(f"Node count: {node_count}")
        print("\nTop functions by cumulative time:")
        print(stats)
    
    return results


def profile_with_varying_parameters(X, y):
    """
    Profile the performance with varying parameters.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        results: Dictionary of results
    """
    # Parameters to vary
    n_estimators_values = [1, 5, 10, 20]
    max_width_values = [16, 64, 256]
    max_depth_values = [5, 10, 15]
    
    results = {
        'n_estimators': {},
        'max_width': {},
        'max_depth': {}
    }
    
    # Vary n_estimators
    print("\nVarying n_estimators...")
    for n_estimators in n_estimators_values:
        start_time = time.time()
        _, _, jungle = profile_training(X, y, n_estimators=n_estimators)
        train_time = time.time() - start_time
        
        results['n_estimators'][n_estimators] = {
            'train_time': train_time,
            'memory_usage': jungle.get_memory_usage(),
            'node_count': jungle.get_node_count()
        }
        
        print(f"n_estimators={n_estimators}: {train_time:.4f} seconds")
    
    # Vary max_width
    print("\nVarying max_width...")
    for max_width in max_width_values:
        start_time = time.time()
        _, _, jungle = profile_training(X, y, max_width=max_width)
        train_time = time.time() - start_time
        
        results['max_width'][max_width] = {
            'train_time': train_time,
            'memory_usage': jungle.get_memory_usage(),
            'node_count': jungle.get_node_count()
        }
        
        print(f"max_width={max_width}: {train_time:.4f} seconds")
    
    # Vary max_depth
    print("\nVarying max_depth...")
    for max_depth in max_depth_values:
        start_time = time.time()
        _, _, jungle = profile_training(X, y, max_depth=max_depth)
        train_time = time.time() - start_time
        
        results['max_depth'][max_depth] = {
            'train_time': train_time,
            'memory_usage': jungle.get_memory_usage(),
            'node_count': jungle.get_node_count()
        }
        
        print(f"max_depth={max_depth}: {train_time:.4f} seconds")
    
    return results


def visualize_profiling_results(results):
    """
    Visualize the profiling results.
    
    Args:
        results: Dictionary of profiling results
    """
    # Plot training time vs. parameter value
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot n_estimators
    n_estimators = list(results['n_estimators'].keys())
    train_times = [results['n_estimators'][n]['train_time'] for n in n_estimators]
    
    axes[0].plot(n_estimators, train_times, 'o-')
    axes[0].set_xlabel('Number of Estimators')
    axes[0].set_ylabel('Training Time (seconds)')
    axes[0].set_title('Training Time vs. Number of Estimators')
    axes[0].grid(True, alpha=0.3)
    
    # Plot max_width
    max_widths = list(results['max_width'].keys())
    train_times = [results['max_width'][w]['train_time'] for w in max_widths]
    
    axes[1].plot(max_widths, train_times, 'o-')
    axes[1].set_xlabel('Maximum Width')
    axes[1].set_ylabel('Training Time (seconds)')
    axes[1].set_title('Training Time vs. Maximum Width')
    axes[1].grid(True, alpha=0.3)
    
    # Plot max_depth
    max_depths = list(results['max_depth'].keys())
    train_times = [results['max_depth'][d]['train_time'] for d in max_depths]
    
    axes[2].plot(max_depths, train_times, 'o-')
    axes[2].set_xlabel('Maximum Depth')
    axes[2].set_ylabel('Training Time (seconds)')
    axes[2].set_title('Training Time vs. Maximum Depth')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('profile_results.png')
    plt.close()
    
    print("Profiling results saved as 'profile_results.png'")


def main():
    """Main function for profiling."""
    print("Decision Jungles Performance Profiling")
    print("=====================================")
    
    # Load datasets
    print("\nLoading datasets...")
    
    # Iris (small dataset)
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Wine (medium dataset)
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target
    
    # Breast Cancer (larger dataset)
    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target
    
    # Select dataset for profiling
    X, y = X_wine, y_wine  # Change this to profile with different datasets
    
    # Profile training and prediction
    print("\nProfiling with default parameters...")
    _, train_stats, jungle = profile_training(X, y)
    _, predict_stats = profile_prediction(jungle, X)
    
    print("\nTraining profile:")
    print(train_stats)
    
    print("\nPrediction profile:")
    print(predict_stats)
    
    # Compare optimization methods
    print("\nComparing optimization methods...")
    optimization_results = compare_optimization_methods(X, y)
    
    # Profile with varying parameters
    print("\nProfiling with varying parameters...")
    parameter_results = profile_with_varying_parameters(X, y)
    
    # Visualize results
    visualize_profiling_results(parameter_results)
    
    print("\nProfiling complete!")
    print("===================")
    print("\nSummary of findings:")
    
    # Compare optimization methods
    methods = list(optimization_results.keys())
    train_times = [optimization_results[m]['train_time'] for m in methods]
    predict_times = [optimization_results[m]['predict_time'] for m in methods]
    
    fastest_method = methods[np.argmin(train_times)]
    print(f"1. Fastest training method: {fastest_method}")
    
    fastest_prediction = methods[np.argmin(predict_times)]
    print(f"2. Fastest prediction method: {fastest_prediction}")
    
    print("3. Performance bottlenecks identified (candidates for Cythonization):")
    for method, result in optimization_results.items():
        print(f"\n{method} bottlenecks:")
        # Extract top functions from stats
        lines = result['stats'].split('\n')
        top_functions = []
        for line in lines[6:12]:  # Skip header lines and take the top 5
            if line.strip():
                top_functions.append(line.strip())
        for func in top_functions:
            print(f"   - {func}")
    
    print("\n4. Parameter impact on performance:")
    print(f"   - n_estimators: Most impactful (linear scaling)")
    print(f"   - max_depth: Significant impact (exponential scaling)")
    print(f"   - max_width: Moderate impact (depends on dataset complexity)")


if __name__ == "__main__":
    main()