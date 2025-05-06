"""
Example demonstrating the use of DecisionJungleRegressor.

This example compares DecisionJungleRegressor with scikit-learn's RandomForestRegressor
on several regression datasets, measuring prediction performance, memory usage,
and training/prediction time.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.datasets import load_diabetes, fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_jungles import DecisionJungleRegressor


def memory_usage_mb(regressor):
    """
    Calculate memory usage in MB, with appropriate handling for both estimators.
    
    For RandomForestRegressor, use a simple heuristic based on number of nodes.
    For DecisionJungleRegressor, use the built-in get_memory_usage() method.
    """
    if hasattr(regressor, 'get_memory_usage'):
        # DecisionJungleRegressor
        return regressor.get_memory_usage() / (1024 * 1024)
    else:
        # RandomForestRegressor - estimate based on node count
        total_nodes = 0
        for estimator in regressor.estimators_:
            total_nodes += estimator.tree_.node_count
        # Each node uses ~60 bytes on average (feature, threshold, children, values, etc.)
        return total_nodes * 60 / (1024 * 1024)


def node_count(regressor):
    """Get total node count for both regressor types."""
    if hasattr(regressor, 'get_node_count'):
        # DecisionJungleRegressor
        return regressor.get_node_count()
    else:
        # RandomForestRegressor
        total_nodes = 0
        for estimator in regressor.estimators_:
            total_nodes += estimator.tree_.node_count
        return total_nodes


def run_experiment(X, y, dataset_name):
    """
    Run experiment comparing DecisionJungleRegressor to RandomForestRegressor.
    
    Returns a dictionary of metrics for comparison.
    """
    results = {
        'Dataset': dataset_name,
        'Samples': X.shape[0],
        'Features': X.shape[1]
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Common parameters for both regressors
    common_params = {
        'n_estimators': 50,
        'random_state': 42
    }
    
    # Initialize regressors
    jungle_regressor = DecisionJungleRegressor(max_width=128, **common_params)
    forest_regressor = RandomForestRegressor(**common_params)
    
    # Train and evaluate models
    regressors = [
        ('Decision Jungle', jungle_regressor),
        ('Random Forest', forest_regressor)
    ]
    
    for name, regressor in regressors:
        print(f"Training {name} on {dataset_name}...")
        
        # Train
        start_time = time()
        regressor.fit(X_train, y_train)
        train_time = time() - start_time
        
        # Predict
        start_time = time()
        y_pred = regressor.predict(X_test)
        pred_time = time() - start_time
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mem_usage = memory_usage_mb(regressor)
        nodes = node_count(regressor)
        
        # Store results
        results[f'{name} MSE'] = mse
        results[f'{name} R²'] = r2
        results[f'{name} Training Time (s)'] = train_time
        results[f'{name} Prediction Time (s)'] = pred_time
        results[f'{name} Memory (MB)'] = mem_usage
        results[f'{name} Node Count'] = nodes
        
        print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
        print(f"  Training Time: {train_time:.4f}s, Prediction Time: {pred_time:.4f}s")
        print(f"  Memory Usage: {mem_usage:.2f} MB, Nodes: {nodes}")
        print()
    
    # Calculated relative metrics
    results['Memory Reduction (%)'] = 100 * (1 - results['Decision Jungle Memory (MB)'] / results['Random Forest Memory (MB)'])
    results['Node Count Reduction (%)'] = 100 * (1 - results['Decision Jungle Node Count'] / results['Random Forest Node Count'])
    results['R² Ratio (DJ/RF)'] = results['Decision Jungle R²'] / results['Random Forest R²']
    
    # Print comparison
    print(f"Comparison ({dataset_name}):")
    print(f"  Memory Reduction: {results['Memory Reduction (%)']:.2f}%")
    print(f"  Node Count Reduction: {results['Node Count Reduction (%)']:.2f}%")
    print(f"  R² Ratio (DJ/RF): {results['R² Ratio (DJ/RF)']:.4f}")
    print("-" * 80)
    
    return results


def main():
    """Run the main experiment on multiple datasets."""
    # Define datasets to use
    datasets = [
        ('Diabetes', load_diabetes(return_X_y=True)),
        ('California Housing', fetch_california_housing(return_X_y=True)),
        ('Synthetic (Low Noise)', make_regression(n_samples=1000, n_features=20, 
                                                 n_informative=10, noise=0.1, random_state=42)),
        ('Synthetic (High Noise)', make_regression(n_samples=1000, n_features=20, 
                                                  n_informative=10, noise=0.5, random_state=42))
    ]
    
    # Run experiments
    results = []
    for name, (X, y) in datasets:
        result = run_experiment(X, y, name)
        results.append(result)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Save to CSV
    summary_df.to_csv('regression_comparison_results.csv', index=False)
    print(f"Results saved to regression_comparison_results.csv")
    
    # Create memory usage comparison chart
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    jungle_mem = [r['Decision Jungle Memory (MB)'] for r in results]
    forest_mem = [r['Random Forest Memory (MB)'] for r in results]
    
    plt.bar(x - width/2, jungle_mem, width, label='Decision Jungle')
    plt.bar(x + width/2, forest_mem, width, label='Random Forest')
    
    plt.xlabel('Dataset')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.xticks(x, [r['Dataset'] for r in results], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('regression_memory_comparison.png')
    
    # Create R² comparison chart
    plt.figure(figsize=(12, 8))
    
    jungle_r2 = [r['Decision Jungle R²'] for r in results]
    forest_r2 = [r['Random Forest R²'] for r in results]
    
    plt.bar(x - width/2, jungle_r2, width, label='Decision Jungle')
    plt.bar(x + width/2, forest_r2, width, label='Random Forest')
    
    plt.xlabel('Dataset')
    plt.ylabel('R² Score')
    plt.title('Prediction Performance Comparison')
    plt.xticks(x, [r['Dataset'] for r in results], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('regression_performance_comparison.png')
    
    print("Charts saved as regression_memory_comparison.png and regression_performance_comparison.png")


if __name__ == "__main__":
    main()