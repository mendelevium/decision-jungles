"""
Example script demonstrating benchmarking utilities for Decision Jungles.

This script performs benchmarks to compare optimized vs standard implementations,
scaling behavior, and hyperparameter impact.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import our DecisionJungleClassifier
from decision_jungles import DecisionJungleClassifier
from decision_jungles.utils.benchmarking import (
    benchmark_optimization,
    benchmark_scaling,
    benchmark_hyperparameters
)


def main():
    # Set up smaller benchmarks for demonstration purposes
    small_benchmark = True
    
    # Load datasets
    print("Loading datasets...")
    iris = load_iris(return_X_y=True)
    breast_cancer = load_breast_cancer(return_X_y=True)
    
    # Example 1: Compare optimized vs standard implementation
    print("\n===== Example 1: Standard vs Optimized Implementation =====")
    print("Using Iris dataset")
    
    # Run benchmark with fewer iterations for demonstration
    results = benchmark_optimization(
        X=iris[0],
        y=iris[1],
        jungle_class=DecisionJungleClassifier,
        n_repeats=2 if small_benchmark else 5,
        n_estimators=3 if small_benchmark else 10,
        max_width=32,
        max_depth=5,
        random_state=42,
    )
    
    # Display results summary
    print("\nOptimization Benchmark Results:")
    print(results.groupby("implementation").agg({
        "train_time": ["mean"],
        "predict_time": ["mean"],
        "accuracy": ["mean"],
    }).round(4))
    
    # Create summary plot
    plt.figure(figsize=(12, 4))
    
    # Training time comparison
    plt.subplot(1, 3, 1)
    mean_times = results.groupby("implementation")["train_time"].mean()
    plt.bar(mean_times.index, mean_times.values)
    plt.ylabel("Training Time (s)")
    plt.title("Training Time Comparison")
    plt.xticks(rotation=45)
    
    # Prediction time comparison
    plt.subplot(1, 3, 2)
    mean_times = results.groupby("implementation")["predict_time"].mean()
    plt.bar(mean_times.index, mean_times.values)
    plt.ylabel("Prediction Time (s)")
    plt.title("Prediction Time Comparison")
    plt.xticks(rotation=45)
    
    # Accuracy comparison
    plt.subplot(1, 3, 3)
    mean_acc = results.groupby("implementation")["accuracy"].mean()
    plt.bar(mean_acc.index, mean_acc.values)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("optimization_benchmark.png")
    plt.close()
    
    print("Optimization benchmark plot saved as 'optimization_benchmark.png'")
    
    # Example 2: Scaling behavior
    if not small_benchmark:
        print("\n===== Example 2: Scaling Behavior =====")
        
        # For full benchmark, use larger datasets and parameter ranges
        scaling_results = benchmark_scaling(
            jungle_class=DecisionJungleClassifier,
            sizes=[500, 1000, 5000],  # Smaller sizes for demonstration
            features=[5, 10, 20],     # Fewer features for demonstration
            n_estimators=5,
            max_width=32,
            max_depth=5,
            use_optimized=True,       # Use optimized implementation
            random_state=42,
            plot=True,
            save_path="scaling_benchmark.png",
        )
        
        print("Scaling benchmark plot saved as 'scaling_benchmark.png'")
    else:
        print("\n===== Example 2: Scaling Behavior =====")
        print("Skipping full scaling benchmark for demonstration purposes.")
        
        # Run a very small scaling benchmark for demonstration
        scaling_results = benchmark_scaling(
            jungle_class=DecisionJungleClassifier,
            sizes=[500, 1000],        # Very small dataset sizes
            features=[5, 10],         # Very few features
            n_estimators=3,           # Few estimators
            max_width=16,
            max_depth=4,
            use_optimized=True,
            random_state=42,
            plot=True,
            save_path="scaling_benchmark.png",
        )
        
        print("Scaling benchmark plot saved as 'scaling_benchmark.png'")
    
    # Example 3: Hyperparameter impact
    print("\n===== Example 3: Hyperparameter Impact =====")
    print("Using Breast Cancer dataset")
    
    # Set smaller parameter ranges for demonstration
    if small_benchmark:
        n_estimators_range = [1, 5, 10]
        max_width_range = [16, 64]
        max_depth_range = [5, None]
    else:
        n_estimators_range = [1, 5, 10, 20]
        max_width_range = [16, 64, 256]
        max_depth_range = [5, 10, None]
    
    # Run hyperparameter benchmark
    hyperparameter_results = benchmark_hyperparameters(
        X=breast_cancer[0],
        y=breast_cancer[1],
        jungle_class=DecisionJungleClassifier,
        n_estimators_range=n_estimators_range,
        max_width_range=max_width_range,
        max_depth_range=max_depth_range,
        use_optimized=True,           # Use optimized implementation
        random_state=42,
        plot=True,
        save_path="hyperparameter_benchmark.png",
    )
    
    print("Hyperparameter benchmark plot saved as 'hyperparameter_benchmark.png'")


if __name__ == "__main__":
    main()