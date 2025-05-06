"""
Comparison of LSearch and ClusterSearch optimization methods for Decision Jungles.

This example compares the two optimization methods for Decision Jungles:
1. LSearch: The default optimization method that alternates between feature
   and branch optimization steps.
2. ClusterSearch: An alternative method that builds temporary nodes and then
   clusters them based on class distribution similarity.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Import our modules
from decision_jungles import DecisionJungleClassifier


def compare_optimization_methods(X, y, dataset_name, max_depth=10,
                                widths=[32, 64, 128, 256], n_estimators=5,
                                n_trials=3, random_state=42):
    """
    Compare LSearch and ClusterSearch optimization methods.
    
    Args:
        X: Feature matrix
        y: Target labels
        dataset_name: Name of the dataset for reporting
        max_depth: Maximum depth of the DAGs
        widths: List of max_width values to try
        n_estimators: Number of DAGs in the ensemble
        n_trials: Number of trials to average results
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary of results
    """
    print(f"\nComparing optimization methods on {dataset_name} dataset...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    results = []
    
    for width in widths:
        # Run multiple trials to get more stable results
        for trial in range(n_trials):
            rs = random_state + trial
            
            # Train with LSearch
            print(f"Training with LSearch, width={width}, trial={trial+1}...")
            start_time = time.time()
            lsearch_jungle = DecisionJungleClassifier(
                n_estimators=n_estimators,
                max_width=width,
                max_depth=max_depth,
                random_state=rs,
                optimization_method="lsearch"
            )
            lsearch_jungle.fit(X_train, y_train)
            lsearch_time = time.time() - start_time
            
            # Evaluate
            y_pred = lsearch_jungle.predict(X_test)
            lsearch_accuracy = accuracy_score(y_test, y_pred)
            lsearch_nodes = lsearch_jungle.get_node_count()
            
            # Train with ClusterSearch
            print(f"Training with ClusterSearch, width={width}, trial={trial+1}...")
            start_time = time.time()
            clustersearch_jungle = DecisionJungleClassifier(
                n_estimators=n_estimators,
                max_width=width,
                max_depth=max_depth,
                random_state=rs,
                optimization_method="clustersearch"
            )
            clustersearch_jungle.fit(X_train, y_train)
            clustersearch_time = time.time() - start_time
            
            # Evaluate
            y_pred = clustersearch_jungle.predict(X_test)
            clustersearch_accuracy = accuracy_score(y_test, y_pred)
            clustersearch_nodes = clustersearch_jungle.get_node_count()
            
            # Store results
            results.append({
                'width': width,
                'trial': trial,
                'lsearch_accuracy': lsearch_accuracy,
                'lsearch_nodes': lsearch_nodes,
                'lsearch_time': lsearch_time,
                'clustersearch_accuracy': clustersearch_accuracy,
                'clustersearch_nodes': clustersearch_nodes,
                'clustersearch_time': clustersearch_time
            })
    
    # Aggregate results
    aggregated = {}
    for width in widths:
        width_results = [r for r in results if r['width'] == width]
        aggregated[width] = {
            'lsearch_accuracy': np.mean([r['lsearch_accuracy'] for r in width_results]),
            'lsearch_accuracy_std': np.std([r['lsearch_accuracy'] for r in width_results]),
            'lsearch_nodes': np.mean([r['lsearch_nodes'] for r in width_results]),
            'lsearch_time': np.mean([r['lsearch_time'] for r in width_results]),
            'clustersearch_accuracy': np.mean([r['clustersearch_accuracy'] for r in width_results]),
            'clustersearch_accuracy_std': np.std([r['clustersearch_accuracy'] for r in width_results]),
            'clustersearch_nodes': np.mean([r['clustersearch_nodes'] for r in width_results]),
            'clustersearch_time': np.mean([r['clustersearch_time'] for r in width_results])
        }
    
    return {
        'dataset_name': dataset_name,
        'widths': widths,
        'results': results,
        'aggregated': aggregated
    }


def plot_comparison(results, output_prefix):
    """
    Plot comparison results between optimization methods.
    
    Args:
        results: Dictionary of results from compare_optimization_methods
        output_prefix: Prefix for output files
    """
    dataset_name = results['dataset_name']
    widths = results['widths']
    aggregated = results['aggregated']
    
    # Setup data for plotting
    lsearch_acc = [aggregated[w]['lsearch_accuracy'] for w in widths]
    lsearch_acc_std = [aggregated[w]['lsearch_accuracy_std'] for w in widths]
    clustersearch_acc = [aggregated[w]['clustersearch_accuracy'] for w in widths]
    clustersearch_acc_std = [aggregated[w]['clustersearch_accuracy_std'] for w in widths]
    
    lsearch_nodes = [aggregated[w]['lsearch_nodes'] for w in widths]
    clustersearch_nodes = [aggregated[w]['clustersearch_nodes'] for w in widths]
    
    lsearch_time = [aggregated[w]['lsearch_time'] for w in widths]
    clustersearch_time = [aggregated[w]['clustersearch_time'] for w in widths]
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.errorbar(widths, lsearch_acc, yerr=lsearch_acc_std, 
                marker='o', label='LSearch', capsize=5)
    plt.errorbar(widths, clustersearch_acc, yerr=clustersearch_acc_std,
                marker='s', label='ClusterSearch', capsize=5)
    plt.xlabel('Max Width (M parameter)')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Comparison - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_prefix}_accuracy.png')
    plt.close()
    
    # Plot node count comparison
    plt.figure(figsize=(10, 6))
    plt.plot(widths, lsearch_nodes, 'o-', label='LSearch')
    plt.plot(widths, clustersearch_nodes, 's-', label='ClusterSearch')
    plt.xlabel('Max Width (M parameter)')
    plt.ylabel('Number of Nodes')
    plt.title(f'Node Count Comparison - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_prefix}_nodes.png')
    plt.close()
    
    # Plot training time comparison
    plt.figure(figsize=(10, 6))
    plt.plot(widths, lsearch_time, 'o-', label='LSearch')
    plt.plot(widths, clustersearch_time, 's-', label='ClusterSearch')
    plt.xlabel('Max Width (M parameter)')
    plt.ylabel('Training Time (seconds)')
    plt.title(f'Training Time Comparison - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_prefix}_time.png')
    plt.close()
    
    # Print summary
    print(f"\nResults for {dataset_name}:")
    print("Accuracy Comparison:")
    for w in widths:
        ls_acc = aggregated[w]['lsearch_accuracy']
        cs_acc = aggregated[w]['clustersearch_accuracy']
        diff = ls_acc - cs_acc
        better = "LSearch" if diff > 0 else "ClusterSearch"
        print(f"  Width {w}: LSearch = {ls_acc:.4f}, ClusterSearch = {cs_acc:.4f}, Diff = {abs(diff):.4f} ({better} better)")
    
    print("\nNode Count Comparison:")
    for w in widths:
        ls_nodes = aggregated[w]['lsearch_nodes']
        cs_nodes = aggregated[w]['clustersearch_nodes']
        ratio = ls_nodes / cs_nodes if cs_nodes > 0 else float('inf')
        better = "ClusterSearch" if ratio > 1 else "LSearch"
        print(f"  Width {w}: LSearch = {ls_nodes:.0f}, ClusterSearch = {cs_nodes:.0f}, Ratio = {ratio:.2f}x ({better} more compact)")
    
    print("\nTraining Time Comparison:")
    for w in widths:
        ls_time = aggregated[w]['lsearch_time']
        cs_time = aggregated[w]['clustersearch_time']
        ratio = ls_time / cs_time if cs_time > 0 else float('inf')
        better = "ClusterSearch" if ratio > 1 else "LSearch"
        print(f"  Width {w}: LSearch = {ls_time:.2f}s, ClusterSearch = {cs_time:.2f}s, Ratio = {ratio:.2f}x ({better} faster)")


def main():
    """Main function to run the comparison."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Compare on smaller datasets with multiple trials for stability
    print("=== Iris Dataset ===")
    X, y = load_iris(return_X_y=True)
    iris_results = compare_optimization_methods(
        X, y, "Iris", max_depth=5, widths=[8, 16, 32, 64], n_trials=3
    )
    plot_comparison(iris_results, "iris_optimization")
    
    print("\n=== Breast Cancer Dataset ===")
    X, y = load_breast_cancer(return_X_y=True)
    cancer_results = compare_optimization_methods(
        X, y, "Breast Cancer", max_depth=8, widths=[16, 32, 64, 128], n_trials=3
    )
    plot_comparison(cancer_results, "cancer_optimization")
    
    # Compare on a larger, more complex dataset with fewer trials
    print("\n=== Digits Dataset ===")
    X, y = load_digits(return_X_y=True)
    digits_results = compare_optimization_methods(
        X, y, "Digits", max_depth=10, widths=[32, 64, 128, 256], n_trials=2
    )
    plot_comparison(digits_results, "digits_optimization")
    
    # Print overall summary
    print("\n=== Overall Summary ===")
    print("Comparison between LSearch and ClusterSearch:")
    print("1. LSearch typically achieves better accuracy.")
    print("2. ClusterSearch may create more compact models (fewer nodes) in some cases.")
    print("3. Training time varies by dataset, but LSearch is often more efficient.")
    print("\nRecommended usage:")
    print("- For best accuracy: Use LSearch (default)")
    print("- For extremely memory-constrained applications: Try ClusterSearch")
    print("- For experimentation: Compare both methods on your specific dataset")


if __name__ == "__main__":
    main()
