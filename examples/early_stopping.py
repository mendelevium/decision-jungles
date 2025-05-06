#!/usr/bin/env python
"""
Example of using early stopping with Decision Jungles.

This example demonstrates how to use the early stopping functionality
to prevent overfitting and reduce training time.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Import our Decision Jungle classifier
from decision_jungles import DecisionJungleClassifier


def plot_early_stopping_comparison(with_early_stopping_results, without_early_stopping_results):
    """
    Plot comparison of models with and without early stopping.
    
    Args:
        with_early_stopping_results: Results from the model with early stopping
        without_early_stopping_results: Results from the model without early stopping
    """
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot accuracies
    axes[0, 0].bar(['With Early Stopping', 'Without Early Stopping'], 
                  [with_early_stopping_results['test_accuracy'], 
                   without_early_stopping_results['test_accuracy']],
                  color=['green', 'blue'])
    axes[0, 0].set_ylim(0.7, 1.0)  # Adjust as needed
    axes[0, 0].set_title('Test Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    
    # Plot training times
    axes[0, 1].bar(['With Early Stopping', 'Without Early Stopping'], 
                  [with_early_stopping_results['training_time'], 
                   without_early_stopping_results['training_time']],
                  color=['green', 'blue'])
    axes[0, 1].set_title('Training Time (seconds)')
    axes[0, 1].set_ylabel('Seconds')
    
    # Plot number of nodes
    axes[1, 0].bar(['With Early Stopping', 'Without Early Stopping'], 
                  [with_early_stopping_results['node_count'], 
                   without_early_stopping_results['node_count']],
                  color=['green', 'blue'])
    axes[1, 0].set_title('Model Size (Number of Nodes)')
    axes[1, 0].set_ylabel('Nodes')
    
    # Plot depths
    axes[1, 1].bar(['With Early Stopping', 'Without Early Stopping'], 
                  [with_early_stopping_results['max_depth'], 
                   without_early_stopping_results['max_depth']],
                  color=['green', 'blue'])
    axes[1, 1].set_title('Maximum Depth')
    axes[1, 1].set_ylabel('Depth')
    
    plt.tight_layout()
    plt.savefig('early_stopping_comparison.png')
    plt.close()
    print("Comparison plot saved as 'early_stopping_comparison.png'")


def plot_validation_scores(validation_scores):
    """
    Plot validation scores across iterations.
    
    Args:
        validation_scores: List of validation scores from each DAG
    """
    plt.figure(figsize=(10, 6))
    
    for i, scores in enumerate(validation_scores):
        # Plot scores for each DAG
        iterations = np.arange(1, len(scores) + 1)
        plt.plot(iterations, scores, label=f'DAG {i+1}')
    
    plt.xlabel('Iteration (Depth Level)')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Scores During Training')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('validation_scores.png')
    plt.close()
    print("Validation scores plot saved as 'validation_scores.png'")


def main():
    """Main function."""
    print("Early Stopping Example for Decision Jungles")
    print("==========================================")
    
    # Generate a dataset with some noise for better early stopping demonstration
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=3,
        random_state=42
    )
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Train a model with early stopping
    print("\nTraining Decision Jungle with early stopping...")
    start_time = time.time()
    jungle_with_es = DecisionJungleClassifier(
        n_estimators=5,
        max_width=64,
        max_depth=20,  # Intentionally deep to demonstrate early stopping
        min_samples_split=5,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=2,
        tol=0.001
    )
    jungle_with_es.fit(X_train, y_train)
    with_es_time = time.time() - start_time
    
    # Get validation scores from DAGs
    validation_scores = [dag.validation_scores for dag in jungle_with_es.dags_]
    
    # Train a model without early stopping for comparison
    print("\nTraining Decision Jungle without early stopping...")
    start_time = time.time()
    jungle_without_es = DecisionJungleClassifier(
        n_estimators=5,
        max_width=64,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    jungle_without_es.fit(X_train, y_train)
    without_es_time = time.time() - start_time
    
    # Evaluate both models
    with_es_pred = jungle_with_es.predict(X_test)
    without_es_pred = jungle_without_es.predict(X_test)
    
    with_es_accuracy = accuracy_score(y_test, with_es_pred)
    without_es_accuracy = accuracy_score(y_test, without_es_pred)
    
    # Collect results
    with_es_results = {
        'test_accuracy': with_es_accuracy,
        'training_time': with_es_time,
        'node_count': jungle_with_es.get_node_count(),
        'max_depth': jungle_with_es.get_max_depth(),
        'stopped_early': jungle_with_es.stopped_early_
    }
    
    without_es_results = {
        'test_accuracy': without_es_accuracy,
        'training_time': without_es_time,
        'node_count': jungle_without_es.get_node_count(),
        'max_depth': jungle_without_es.get_max_depth()
    }
    
    # Print results
    print("\n=== Results ===")
    print("\nModel with early stopping:")
    print(f"  Test accuracy: {with_es_accuracy:.4f}")
    print(f"  Training time: {with_es_time:.2f} seconds")
    print(f"  Node count: {with_es_results['node_count']}")
    print(f"  Max depth: {with_es_results['max_depth']}")
    print(f"  Stopped early: {with_es_results['stopped_early']}")
    
    print("\nModel without early stopping:")
    print(f"  Test accuracy: {without_es_accuracy:.4f}")
    print(f"  Training time: {without_es_time:.2f} seconds")
    print(f"  Node count: {without_es_results['node_count']}")
    print(f"  Max depth: {without_es_results['max_depth']}")
    
    # Calculate savings
    time_saving = without_es_time / with_es_time if with_es_time > 0 else 0
    node_reduction = without_es_results['node_count'] / with_es_results['node_count'] if with_es_results['node_count'] > 0 else 0
    
    print("\nSavings with early stopping:")
    print(f"  Time: {time_saving:.2f}x faster")
    print(f"  Model size: {node_reduction:.2f}x smaller")
    
    # Create visualizations
    plot_early_stopping_comparison(with_es_results, without_es_results)
    plot_validation_scores(validation_scores)
    
    print("\nEarly stopping benefits:")
    print("1. Reduced training time by stopping when validation performance plateaus")
    print("2. Potentially better generalization by preventing overfitting")
    print("3. Smaller model size with comparable or better performance")
    print("4. Automatic determination of optimal depth for each DAG")


if __name__ == "__main__":
    main()