#!/usr/bin/env python
"""
Hyperparameter Tuning for Decision Jungles using Cross-Validation.

This example demonstrates how to use scikit-learn's GridSearchCV and RandomizedSearchCV
to find optimal hyperparameters for a Decision Jungle model. Cross-validation is used
to evaluate model performance across different hyperparameter combinations.

The example also visualizes the impact of different hyperparameters on model performance
and compares the best Decision Jungle model with a similarly tuned Random Forest.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time
import pandas as pd
from scipy.stats import uniform, randint

# Import our Decision Jungle classifier
from decision_jungles import DecisionJungleClassifier
from decision_jungles.utils.metrics import compare_memory_usage, measure_prediction_time


def grid_search_example(X_train, y_train, X_test, y_test, cv=5):
    """
    Perform grid search to find optimal hyperparameters for Decision Jungle.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        cv: Number of cross-validation folds
        
    Returns:
        Fitted GridSearchCV object and best model
    """
    print("Running Grid Search for Decision Jungle...")
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [5, 10, 20],
        'max_width': [32, 64, 128],
        'max_depth': [5, 10, 15],
        'merging_schedule': ['constant', 'exponential'],
        'min_samples_split': [2, 5, 10]
    }
    
    # Create a Decision Jungle classifier
    jungle = DecisionJungleClassifier(random_state=42)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=jungle,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    # Fit the grid search to the data
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    grid_search_time = time.time() - start_time
    
    # Get the best model
    best_jungle = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_jungle.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print("\nGrid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Time taken: {grid_search_time:.2f} seconds")
    
    # Memory usage
    memory_usage = best_jungle.get_memory_usage()
    node_count = best_jungle.get_node_count()
    print(f"Memory usage: {memory_usage} bytes")
    print(f"Total nodes: {node_count}")
    
    # Visualize and analyze results
    analyze_grid_search_results(grid_search, X_test, y_test)
    
    return grid_search, best_jungle


def randomized_search_example(X_train, y_train, X_test, y_test, n_iter=20, cv=5):
    """
    Perform randomized search to find good hyperparameters for Decision Jungle.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        n_iter: Number of parameter settings to sample
        cv: Number of cross-validation folds
        
    Returns:
        Fitted RandomizedSearchCV object and best model
    """
    print("\nRunning Randomized Search for Decision Jungle...")
    
    # Define the parameter distributions
    param_dist = {
        'n_estimators': randint(5, 30),
        'max_width': randint(16, 256),
        'max_depth': randint(3, 20),
        'merging_schedule': ['constant', 'exponential', 'kinect'],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7]
    }
    
    # Create a Decision Jungle classifier
    jungle = DecisionJungleClassifier(random_state=42)
    
    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=jungle,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    # Fit the random search to the data
    start_time = time.time()
    random_search.fit(X_train, y_train)
    random_search_time = time.time() - start_time
    
    # Get the best model
    best_jungle = random_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_jungle.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print("\nRandomized Search Results:")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Time taken: {random_search_time:.2f} seconds")
    
    # Memory usage
    memory_usage = best_jungle.get_memory_usage()
    node_count = best_jungle.get_node_count()
    print(f"Memory usage: {memory_usage} bytes")
    print(f"Total nodes: {node_count}")
    
    return random_search, best_jungle


def analyze_grid_search_results(grid_search, X_test, y_test):
    """
    Analyze and visualize grid search results.
    
    Args:
        grid_search: Fitted GridSearchCV object
        X_test: Test features
        y_test: Test labels
    """
    # Convert results to a DataFrame
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Parameters to analyze
    params_to_analyze = ['param_n_estimators', 'param_max_width', 'param_max_depth']
    
    # Create plots for each parameter
    fig, axes = plt.subplots(1, len(params_to_analyze), figsize=(15, 5))
    
    for i, param in enumerate(params_to_analyze):
        # Get unique values of the parameter
        unique_values = results[param].unique()
        unique_values = [str(val) for val in unique_values]
        
        # Calculate mean scores for each value
        mean_scores = []
        std_scores = []
        
        for val in unique_values:
            # Filter results for this parameter value
            param_results = results[results[param].astype(str) == val]
            mean_scores.append(param_results['mean_test_score'].mean())
            std_scores.append(param_results['mean_test_score'].std())
        
        # Plot
        axes[i].errorbar(unique_values, mean_scores, yerr=std_scores, fmt='o-')
        axes[i].set_xlabel(param.replace('param_', ''))
        axes[i].set_ylabel('Mean CV Accuracy')
        axes[i].set_title(f'Effect of {param.replace("param_", "")}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grid_search_params_analysis.png')
    plt.close()
    print("Grid search parameter analysis saved as 'grid_search_params_analysis.png'")
    
    # Analyze the effect of merging schedule and min_samples_split
    schedules = results['param_merging_schedule'].unique()
    min_samples = results['param_min_samples_split'].unique()
    
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    heatmap_data = np.zeros((len(schedules), len(min_samples)))
    
    for i, schedule in enumerate(schedules):
        for j, ms in enumerate(min_samples):
            # Filter results for this combination
            combo_results = results[
                (results['param_merging_schedule'] == schedule) & 
                (results['param_min_samples_split'] == ms)
            ]
            heatmap_data[i, j] = combo_results['mean_test_score'].mean()
    
    plt.imshow(heatmap_data, cmap='viridis')
    plt.colorbar(label='Mean CV Accuracy')
    plt.xticks(np.arange(len(min_samples)), min_samples)
    plt.yticks(np.arange(len(schedules)), schedules)
    plt.xlabel('min_samples_split')
    plt.ylabel('merging_schedule')
    plt.title('Interaction Between merging_schedule and min_samples_split')
    
    # Add text annotations
    for i in range(len(schedules)):
        for j in range(len(min_samples)):
            plt.text(j, i, f"{heatmap_data[i, j]:.3f}", 
                     ha="center", va="center", color="white" if heatmap_data[i, j] < 0.85 else "black")
    
    plt.tight_layout()
    plt.savefig('merging_schedule_interaction.png')
    plt.close()
    print("Merging schedule interaction analysis saved as 'merging_schedule_interaction.png'")


def compare_with_forest(X_train, y_train, X_test, y_test, best_jungle, cv=5):
    """
    Compare the best Decision Jungle model with a tuned Random Forest.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        best_jungle: Best Decision Jungle model from hyperparameter search
        cv: Number of cross-validation folds
    """
    print("\nComparing best Decision Jungle with tuned Random Forest...")
    
    # Use GridSearchCV to find the best Random Forest parameters
    param_grid = {
        'n_estimators': [best_jungle.n_estimators, best_jungle.n_estimators * 2],
        'max_depth': [best_jungle.max_depth, best_jungle.max_depth + 5],
        'min_samples_split': [best_jungle.min_samples_split, 2],
        'min_samples_leaf': [best_jungle.min_samples_leaf, 1],
        'max_features': ['sqrt', 'log2', None]
    }
    
    forest = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=forest,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    best_forest = grid_search.best_estimator_
    
    # Evaluate both models
    jungle_pred = best_jungle.predict(X_test)
    forest_pred = best_forest.predict(X_test)
    
    jungle_accuracy = accuracy_score(y_test, jungle_pred)
    forest_accuracy = accuracy_score(y_test, forest_pred)
    
    # Compare prediction times
    jungle_times = measure_prediction_time(best_jungle, X_test)
    forest_times = measure_prediction_time(best_forest, X_test)
    
    # Compare memory usage
    memory_comparison = compare_memory_usage(best_jungle, best_forest)
    
    # Print results
    print("\n=== Comparison Results ===")
    print("\nBest Parameters:")
    print(f"  Decision Jungle: {best_jungle.get_params()}")
    print(f"  Random Forest: {best_forest.get_params()}")
    
    print("\nAccuracy:")
    print(f"  Decision Jungle: {jungle_accuracy:.4f}")
    print(f"  Random Forest: {forest_accuracy:.4f}")
    
    print("\nPrediction Time (mean):")
    print(f"  Decision Jungle: {jungle_times['mean_time']:.6f} seconds")
    print(f"  Random Forest: {forest_times['mean_time']:.6f} seconds")
    
    print("\nMemory Usage:")
    print(f"  Decision Jungle: {best_jungle.get_memory_usage()} bytes, {best_jungle.get_node_count()} nodes")
    forest_nodes = sum(tree.tree_.node_count for tree in best_forest.estimators_)
    print(f"  Random Forest: Approx. {forest_nodes} nodes")
    print(f"  Memory ratio (Forest/Jungle): {memory_comparison['memory_ratio']:.2f}x")
    
    # Create a visual comparison
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Accuracy
    plt.subplot(2, 2, 1)
    plt.bar(['Decision Jungle', 'Random Forest'], 
            [jungle_accuracy, forest_accuracy],
            color=['green', 'blue'])
    plt.ylim(0.8, 1.0)  # Adjust as needed
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy')
    
    # Plot 2: Model Size
    plt.subplot(2, 2, 2)
    plt.bar(['Decision Jungle', 'Random Forest'], 
            [best_jungle.get_node_count(), forest_nodes],
            color=['green', 'blue'])
    plt.yscale('log')
    plt.title('Model Size (Number of Nodes)')
    plt.ylabel('Nodes (log scale)')
    
    # Plot 3: Prediction Speed
    plt.subplot(2, 2, 3)
    plt.bar(['Decision Jungle', 'Random Forest'], 
            [jungle_times['mean_time'], forest_times['mean_time']],
            color=['green', 'blue'])
    plt.title('Prediction Time')
    plt.ylabel('Time (seconds)')
    
    # Plot 4: Memory Usage Ratio
    plt.subplot(2, 2, 4)
    plt.bar(['Memory Ratio (Forest/Jungle)'], 
            [memory_comparison['memory_ratio']],
            color=['purple'])
    plt.title('Memory Usage Ratio')
    plt.ylabel('Ratio')
    
    plt.tight_layout()
    plt.savefig('jungle_vs_forest_tuned_comparison.png')
    plt.close()
    print("Tuned model comparison saved as 'jungle_vs_forest_tuned_comparison.png'")


def main():
    """Main function."""
    print("Hyperparameter Tuning for Decision Jungles")
    print("==========================================")
    
    # Load a dataset - MNIST digits for a more challenging task
    print("Loading MNIST dataset...")
    digits = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = digits.data, digits.target
    
    # Take a subset for faster tuning
    n_samples = 5000  # Adjust as needed
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_subset = X.iloc[indices].values if hasattr(X, 'iloc') else X[indices]
    y_subset = y.iloc[indices].values if hasattr(y, 'iloc') else y[indices]
    
    print(f"Using {n_samples} samples for hyperparameter tuning")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=0.3, random_state=42
    )
    
    # Run grid search
    grid_search, best_jungle_grid = grid_search_example(X_train, y_train, X_test, y_test)
    
    # Run randomized search
    random_search, best_jungle_random = randomized_search_example(X_train, y_train, X_test, y_test)
    
    # Compare with the best model from grid search
    compare_with_forest(X_train, y_train, X_test, y_test, best_jungle_grid)
    
    print("\nHyperparameter Tuning Complete!")
    print("==================================")
    print("Key findings:")
    print("1. Decision Jungles require careful tuning of max_width and merging_schedule")
    print("2. GridSearchCV provides systematic exploration of the parameter space")
    print("3. RandomizedSearchCV allows exploring a wider range of values more efficiently")
    print("4. Compared to Random Forests, well-tuned Decision Jungles offer:")
    print("   - Comparable accuracy")
    print("   - Significantly reduced memory usage")
    print("   - Different trade-offs in prediction speed")
    print("\nTips for tuning Decision Jungles:")
    print("- Start with n_estimators between 5-20 for efficiency")
    print("- Experiment with max_width values (controls memory vs. accuracy trade-off)")
    print("- Try different merging schedules for different dataset characteristics")
    print("- Consider optimizing for memory usage vs. accuracy based on your application")


if __name__ == "__main__":
    main()