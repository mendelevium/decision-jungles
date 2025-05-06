"""
Comprehensive comparison between Decision Jungles and Random Forests.

This example compares Decision Jungles with Random Forests across different
settings, focusing on memory usage, accuracy, and generalization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Import our modules
from decision_jungles import DecisionJungleClassifier
from decision_jungles.utils.metrics import compare_memory_usage, jaccard_index
from decision_jungles.utils.visualization import (
    plot_memory_comparison, 
    plot_accuracy_vs_nodes,
    plot_accuracy_vs_evaluations
)


def evaluate_models(X, y, dataset_name, max_depths=[5, 10, 15, 20], 
                   widths=[32, 64, 128, 256], n_estimators=5, random_state=42):
    """
    Evaluate jungle and forest models with different parameters.
    
    Args:
        X: Feature matrix
        y: Target labels
        dataset_name: Name of the dataset for reporting
        max_depths: List of max_depth values to try
        widths: List of max_width values to try for jungles
        n_estimators: Number of trees/DAGs in the ensemble
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of results
    """
    print(f"\nEvaluating models on {dataset_name} dataset...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    results = []
    
    # Train and evaluate forests with different depths
    for depth in max_depths:
        print(f"Training Random Forest with max_depth={depth}...")
        
        forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=depth,
            random_state=random_state
        )
        forest.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = forest.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get node count
        node_count = sum(tree.tree_.node_count for tree in forest.estimators_)
        
        # Get max evaluations per sample
        max_evaluations = depth * n_estimators
        
        results.append({
            'name': f'Random Forest (depth={depth})',
            'model_type': 'forest',
            'accuracy': accuracy,
            'nodes': node_count,
            'evaluations': max_evaluations,
            'max_depth': depth
        })
        
    # Train and evaluate jungles with different depths and widths
    for depth in max_depths:
        for width in widths:
            print(f"Training Decision Jungle with max_depth={depth}, max_width={width}...")
            
            jungle = DecisionJungleClassifier(
                n_estimators=n_estimators,
                max_width=width,
                max_depth=depth,
                random_state=random_state
            )
            jungle.fit(X_train, y_train)
            
            # Calculate metrics
            y_pred = jungle.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get node count and max depth
            node_count = jungle.get_node_count()
            actual_max_depth = jungle.get_max_depth()
            
            # Get max evaluations per sample
            max_evaluations = actual_max_depth * n_estimators
            
            results.append({
                'name': f'Decision Jungle (depth={depth}, width={width})',
                'model_type': 'jungle',
                'accuracy': accuracy,
                'nodes': node_count,
                'evaluations': max_evaluations,
                'max_depth': actual_max_depth,
                'max_width': width
            })
    
    return {
        'dataset_name': dataset_name,
        'results': results,
        'X_test': X_test,
        'y_test': y_test
    }


def plot_results(results, output_prefix):
    """
    Plot comparison results.
    
    Args:
        results: Dictionary of results from evaluate_models
        output_prefix: Prefix for output files
    """
    dataset_name = results['dataset_name']
    
    # Filter results for plotting
    forest_results = [r for r in results['results'] if r['model_type'] == 'forest']
    jungle_results = [r for r in results['results'] if r['model_type'] == 'jungle']
    
    # Group jungle results by max_width
    jungle_by_width = {}
    for result in jungle_results:
        width = result['max_width']
        if width not in jungle_by_width:
            jungle_by_width[width] = []
        jungle_by_width[width].append(result)
    
    # Plot accuracy vs nodes
    plt.figure(figsize=(12, 8))
    
    # Plot forest results
    forest_nodes = [r['nodes'] for r in forest_results]
    forest_accuracy = [r['accuracy'] for r in forest_results]
    plt.plot(forest_nodes, forest_accuracy, 'o-', label='Random Forest', color='blue', linewidth=2)
    
    # Plot jungle results for each width
    colors = ['green', 'red', 'purple', 'orange']
    for i, (width, results) in enumerate(jungle_by_width.items()):
        nodes = [r['nodes'] for r in results]
        accuracy = [r['accuracy'] for r in results]
        plt.plot(nodes, accuracy, 'o-', 
                label=f'Decision Jungle (width={width})', 
                color=colors[i % len(colors)], 
                linewidth=2)
    
    plt.xscale('log')
    plt.xlabel('Number of Nodes (log scale)')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Model Size - {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{output_prefix}_accuracy_vs_nodes.png')
    plt.close()
    
    # Plot accuracy vs feature evaluations
    plt.figure(figsize=(12, 8))
    
    # Plot forest results
    forest_evals = [r['evaluations'] for r in forest_results]
    forest_accuracy = [r['accuracy'] for r in forest_results]
    plt.plot(forest_evals, forest_accuracy, 'o-', label='Random Forest', color='blue', linewidth=2)
    
    # Plot jungle results for each width
    for i, (width, results) in enumerate(jungle_by_width.items()):
        evals = [r['evaluations'] for r in results]
        accuracy = [r['accuracy'] for r in results]
        plt.plot(evals, accuracy, 'o-', 
                label=f'Decision Jungle (width={width})', 
                color=colors[i % len(colors)], 
                linewidth=2)
    
    plt.xlabel('Max. Feature Evaluations per Sample')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Computation - {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{output_prefix}_accuracy_vs_evaluations.png')
    plt.close()
    
    # Print best models
    all_results = forest_results + jungle_results
    best_model = max(all_results, key=lambda r: r['accuracy'])
    smallest_decent_model = min(
        [r for r in all_results if r['accuracy'] >= 0.95 * best_model['accuracy']],
        key=lambda r: r['nodes']
    )
    
    print(f"\nResults for {dataset_name}:")
    print(f"Best model: {best_model['name']} with accuracy {best_model['accuracy']:.4f}")
    print(f"Smallest decent model: {smallest_decent_model['name']} with accuracy {smallest_decent_model['accuracy']:.4f}")
    print(f"Memory reduction: {best_model['nodes'] / smallest_decent_model['nodes']:.2f}x")


def main():
    """Main function to run the comparison."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Compare on Iris dataset
    print("=== Iris Dataset ===")
    X, y = load_iris(return_X_y=True)
    iris_results = evaluate_models(X, y, "Iris")
    plot_results(iris_results, "iris")
    
    # Compare on Digits dataset
    print("\n=== Digits Dataset ===")
    X, y = load_digits(return_X_y=True)
    digits_results = evaluate_models(X, y, "Digits")
    plot_results(digits_results, "digits")
    
    # Compare on Breast Cancer dataset
    print("\n=== Breast Cancer Dataset ===")
    X, y = load_breast_cancer(return_X_y=True)
    cancer_results = evaluate_models(X, y, "Breast Cancer")
    plot_results(cancer_results, "breast_cancer")
    
    # Print overall summary
    print("\n=== Overall Summary ===")
    print("Decision Jungles typically achieve:")
    print("1. Reduced memory footprint (fewer nodes)")
    print("2. Comparable or better accuracy than Random Forests")
    print("3. Requires more feature evaluations at test time")
    print("\nRecommended settings:")
    print("- For memory-constrained applications: Use jungles with moderate width (64-128)")
    print("- For best accuracy: Use jungles with larger width (256+)")
    print("- For fastest inference: Use forests with limited depth")


if __name__ == "__main__":
    main()
