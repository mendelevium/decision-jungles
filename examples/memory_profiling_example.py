"""
Example script demonstrating memory profiling utilities for Decision Jungles.

This script compares memory usage between Decision Jungles and Random Forests
with various parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import our DecisionJungleClassifier
from decision_jungles import DecisionJungleClassifier
from decision_jungles.utils.memory_profiling import (
    measure_model_memory,
    memory_usage_vs_accuracy,
    estimate_model_size
)


def main():
    # Load a dataset
    print("Loading breast cancer dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Example 1: Measure memory usage of a single model
    print("\n===== Example 1: Measure memory usage of a single model =====")
    
    # Create and train a Decision Jungle
    jungle = DecisionJungleClassifier(
        n_estimators=10,
        max_width=64,
        max_depth=10,
        random_state=42
    )
    jungle.fit(X_train, y_train)
    
    # Measure memory
    jungle_memory = measure_model_memory(jungle, detailed=False)
    
    # Create and train a Random Forest
    forest = RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        random_state=42
    )
    forest.fit(X_train, y_train)
    
    # Measure memory
    forest_memory = measure_model_memory(forest, detailed=False)
    
    # Display results
    print("\nMemory Usage:")
    print(f"  Decision Jungle: {jungle_memory.get('reported_mb', jungle_memory['rss_mb']):.2f} MB")
    print(f"  Random Forest: {forest_memory['rss_mb']:.2f} MB")
    
    print("\nAccuracy:")
    print(f"  Decision Jungle: {jungle.score(X_test, y_test):.4f}")
    print(f"  Random Forest: {forest.score(X_test, y_test):.4f}")
    
    # Example 2: Estimate memory usage without creating models
    print("\n===== Example 2: Estimate memory usage =====")
    
    # Estimate for different numbers of estimators
    for n in [10, 50, 100]:
        # Estimate memory for Jungle
        jungle_params = {
            'n_estimators': n,
            'max_width': 64,
            'max_depth': 10,
            'n_classes': len(np.unique(y))
        }
        jungle_estimate = estimate_model_size(jungle_params)
        
        print(f"Estimated memory for Decision Jungle with {n} estimators: "
              f"{jungle_estimate['estimated_mb']:.2f} MB "
              f"({jungle_estimate['estimated_nodes']} nodes)")
    
    # Example 3: Compare memory usage vs accuracy for different parameters
    print("\n===== Example 3: Memory vs Accuracy Comparisons =====")
    
    # Create constructor functions
    def jungle_constructor(n_estimators):
        return DecisionJungleClassifier(
            n_estimators=n_estimators,
            max_width=64,
            max_depth=10,
            random_state=42
        )
    
    def forest_constructor(n_estimators):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42
        )
    
    # Compare for different numbers of estimators
    estimator_range = [5, 10, 20]
    
    results = memory_usage_vs_accuracy(
        jungle_constructor=jungle_constructor,
        forest_constructor=forest_constructor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        param_range=estimator_range,
        param_name='Number of Estimators',
        title='Memory Usage vs. Accuracy: Decision Jungle vs. Random Forest',
        save_path='memory_vs_accuracy.png'
    )
    
    # Print summary of results
    print("\nSummary of Memory vs. Accuracy Comparison:")
    print("\nDecision Jungle:")
    for i, n in enumerate(estimator_range):
        print(f"  {n} estimators: {results['jungle']['memory_mb'][i]:.2f} MB, "
              f"accuracy: {results['jungle']['accuracy'][i]:.4f}")
    
    print("\nRandom Forest:")
    for i, n in enumerate(estimator_range):
        print(f"  {n} estimators: {results['forest']['memory_mb'][i]:.2f} MB, "
              f"accuracy: {results['forest']['accuracy'][i]:.4f}")
    
    print("\nPlot saved as 'memory_vs_accuracy.png'")


if __name__ == "__main__":
    main()