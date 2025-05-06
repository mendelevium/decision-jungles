"""
Basic usage example for Decision Jungle classifier.

This example demonstrates how to use the DecisionJungleClassifier
on the Iris dataset, and compares it to scikit-learn's RandomForestClassifier.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Import our DecisionJungleClassifier
from decision_jungles import DecisionJungleClassifier
from decision_jungles.utils.metrics import compare_memory_usage, measure_prediction_time
from decision_jungles.utils.visualization import plot_memory_comparison


def main():
    # Load the Iris dataset
    print("Loading the Iris dataset...")
    X, y = load_iris(return_X_y=True)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Create and train a Decision Jungle classifier
    print("\nTraining Decision Jungle classifier...")
    start_time = time.time()
    jungle = DecisionJungleClassifier(
        n_estimators=10,
        max_width=64,
        max_depth=10,
        random_state=42
    )
    jungle.fit(X_train, y_train)
    jungle_train_time = time.time() - start_time
    
    # Make predictions and evaluate
    y_pred_jungle = jungle.predict(X_test)
    jungle_accuracy = accuracy_score(y_test, y_pred_jungle)
    
    # Get memory usage and node count
    jungle_memory = jungle.get_memory_usage()
    jungle_nodes = jungle.get_node_count()
    
    # Create and train a Random Forest classifier for comparison
    print("Training Random Forest classifier...")
    start_time = time.time()
    forest = RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        random_state=42
    )
    forest.fit(X_train, y_train)
    forest_train_time = time.time() - start_time
    
    # Make predictions and evaluate
    y_pred_forest = forest.predict(X_test)
    forest_accuracy = accuracy_score(y_test, y_pred_forest)
    
    # Get forest memory usage (estimated)
    # For scikit-learn's RandomForestClassifier, we count the total number of nodes
    forest_nodes = sum(tree.tree_.node_count for tree in forest.estimators_)
    
    # Compare prediction times
    jungle_times = measure_prediction_time(jungle, X_test)
    forest_times = measure_prediction_time(forest, X_test)
    
    # Display results
    print("\n=== Results ===")
    print("\nAccuracy:")
    print(f"  Decision Jungle: {jungle_accuracy:.4f}")
    print(f"  Random Forest:   {forest_accuracy:.4f}")
    
    print("\nTraining Time:")
    print(f"  Decision Jungle: {jungle_train_time:.4f} seconds")
    print(f"  Random Forest:   {forest_train_time:.4f} seconds")
    
    print("\nPrediction Time (mean):")
    print(f"  Decision Jungle: {jungle_times['mean_time']:.6f} seconds")
    print(f"  Random Forest:   {forest_times['mean_time']:.6f} seconds")
    
    print("\nMemory Usage:")
    print(f"  Decision Jungle: {jungle_memory} bytes, {jungle_nodes} nodes")
    print(f"  Random Forest:   Approx. {forest_nodes} nodes")
    
    print("\nModel Information:")
    print(f"  Decision Jungle: {len(jungle.dags_)} DAGs, max depth = {jungle.get_max_depth()}")
    print(f"  Random Forest:   {len(forest.estimators_)} trees, max depth = {forest.max_depth}")
    
    # Plot memory comparison
    memory_comparison = compare_memory_usage(jungle, forest)
    print(f"\nMemory ratio (Forest/Jungle): {memory_comparison['memory_ratio']:.2f}x")
    
    # Create a visual comparison
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(['Decision Jungle', 'Random Forest'], 
            [jungle_accuracy, forest_accuracy],
            color=['green', 'blue'])
    plt.ylim(0.8, 1.0)  # Adjust as needed
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.bar(['Decision Jungle', 'Random Forest'], 
            [jungle_nodes, forest_nodes],
            color=['green', 'blue'])
    plt.yscale('log')
    plt.title('Model Size (Number of Nodes)')
    plt.ylabel('Nodes (log scale)')
    
    plt.tight_layout()
    plt.savefig('jungle_vs_forest_comparison.png')
    plt.close()
    
    print("Comparison plot saved as 'jungle_vs_forest_comparison.png'")


if __name__ == "__main__":
    main()
