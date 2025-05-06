#!/usr/bin/env python
"""
Integrating Decision Jungles into scikit-learn Pipelines.

This example demonstrates how to use Decision Jungles within scikit-learn's Pipeline
framework, allowing for seamless preprocessing, feature selection, and model building.
The example also shows how to perform cross-validation and hyperparameter tuning
on complete pipelines containing Decision Jungles.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import time

# Import our Decision Jungle classifier
from decision_jungles import DecisionJungleClassifier
from decision_jungles.utils.metrics import compare_memory_usage


def basic_pipeline_example():
    """
    Basic example of integrating Decision Jungles into a scikit-learn pipeline.
    """
    print("Basic Decision Jungle Pipeline Example")
    print("=====================================")
    
    # Load the wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Create feature names for better interpretability
    feature_names = wine.feature_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create a pipeline with preprocessing and a Decision Jungle classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('jungle', DecisionJungleClassifier(
            n_estimators=10,
            max_width=64,
            max_depth=10,
            random_state=42
        ))
    ])
    
    # Train the pipeline
    print("Training the pipeline...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return pipeline


def feature_selection_pipeline():
    """
    Example of using Decision Jungles with feature selection in a pipeline.
    """
    print("\nDecision Jungle Pipeline with Feature Selection")
    print("=============================================")
    
    # Load the wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Create feature names for better interpretability
    feature_names = wine.feature_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create a pipeline with feature selection
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=5)),  # Select top 5 features
        ('jungle', DecisionJungleClassifier(
            n_estimators=10,
            max_width=64,
            random_state=42
        ))
    ])
    
    # Train the pipeline
    print("Training the pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Get the selected features
    selected_indices = pipeline.named_steps['feature_selection'].get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    print("\nSelected Features:")
    for i, feature in enumerate(selected_features):
        print(f"{i+1}. {feature}")
    
    # Compare with a pipeline without feature selection
    print("\nComparing with a pipeline without feature selection...")
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('jungle', DecisionJungleClassifier(
            n_estimators=10,
            max_width=64,
            random_state=42
        ))
    ])
    
    baseline_pipeline.fit(X_train, y_train)
    baseline_pred = baseline_pipeline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    
    print(f"Baseline accuracy (all features): {baseline_accuracy:.4f}")
    print(f"Feature selection impact: {accuracy - baseline_accuracy:.4f}")
    
    # Get memory usage
    jungle_with_selection = pipeline.named_steps['jungle']
    jungle_baseline = baseline_pipeline.named_steps['jungle']
    
    memory_with_selection = jungle_with_selection.get_memory_usage()
    memory_baseline = jungle_baseline.get_memory_usage()
    
    print(f"\nMemory usage with feature selection: {memory_with_selection} bytes")
    print(f"Memory usage without feature selection: {memory_baseline} bytes")
    print(f"Memory reduction: {memory_baseline / memory_with_selection:.2f}x")
    
    return pipeline


def dimensionality_reduction_pipeline():
    """
    Example of using Decision Jungles with PCA for dimensionality reduction.
    """
    print("\nDecision Jungle Pipeline with PCA")
    print("================================")
    
    # Load the wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create a pipeline with PCA
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=5)),  # Reduce to 5 principal components
        ('jungle', DecisionJungleClassifier(
            n_estimators=10,
            max_width=64,
            random_state=42
        ))
    ])
    
    # Train the pipeline
    print("Training the pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Get explained variance
    pca = pipeline.named_steps['pca']
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()
    
    print(f"\nTotal explained variance: {total_variance:.4f}")
    print("Explained variance by component:")
    for i, var in enumerate(explained_variance):
        print(f"Component {i+1}: {var:.4f} ({var / total_variance:.2%})")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.plot(range(1, len(explained_variance) + 1), 
             np.cumsum(explained_variance), 'r-o', linewidth=2)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.savefig('pca_explained_variance.png')
    plt.close()
    
    print("PCA explained variance plot saved as 'pca_explained_variance.png'")
    
    return pipeline


def pipeline_with_missing_values():
    """
    Example of handling missing values in a pipeline with Decision Jungles.
    """
    print("\nDecision Jungle Pipeline with Missing Value Handling")
    print("==================================================")
    
    # Load the breast cancer dataset
    cancer = fetch_openml(name='breast-cancer-wisconsin', version=1, parser='auto')
    X, y = cancer.data, cancer.target
    
    # Introduce some missing values for demonstration
    rows, cols = X.shape
    n_missing = int(rows * cols * 0.05)  # 5% missing values
    np.random.seed(42)
    
    # Create a copy of X to manipulate
    X_missing = X.copy()
    
    # Randomly set elements to NaN
    for _ in range(n_missing):
        i = np.random.randint(0, rows)
        j = np.random.randint(0, cols)
        X_missing.iloc[i, j] = np.nan if hasattr(X_missing, 'iloc') else np.nan
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_missing, y, test_size=0.3, random_state=42
    )
    
    # Create a pipeline with missing value handling
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('jungle', DecisionJungleClassifier(
            n_estimators=10,
            max_width=64,
            random_state=42
        ))
    ])
    
    # Train the pipeline
    print("Training the pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"Test accuracy with missing values: {accuracy:.4f}")
    
    # Compare with a pipeline trained on complete data
    print("\nComparing with a pipeline trained on complete data...")
    X_train_complete, X_test_complete, y_train_complete, y_test_complete = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('jungle', DecisionJungleClassifier(
            n_estimators=10,
            max_width=64,
            random_state=42
        ))
    ])
    
    baseline_pipeline.fit(X_train_complete, y_train_complete)
    baseline_pred = baseline_pipeline.predict(X_test_complete)
    baseline_accuracy = accuracy_score(y_test_complete, baseline_pred)
    
    print(f"Baseline accuracy (complete data): {baseline_accuracy:.4f}")
    print(f"Accuracy difference: {baseline_accuracy - accuracy:.4f}")
    
    return pipeline


def pipeline_grid_search():
    """
    Example of grid search for hyperparameter tuning on a pipeline with Decision Jungles.
    """
    print("\nGrid Search on Decision Jungle Pipeline")
    print("=====================================")
    
    # Load the wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('jungle', DecisionJungleClassifier(random_state=42))
    ])
    
    # Define the parameter grid for grid search
    param_grid = {
        'feature_selection__k': [5, 8, 10],
        'jungle__n_estimators': [5, 10],
        'jungle__max_width': [32, 64],
        'jungle__max_depth': [5, 10],
        'jungle__merging_schedule': ['constant', 'exponential']
    }
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    print("Running grid search...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    grid_search_time = time.time() - start_time
    
    # Get the best pipeline
    best_pipeline = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print("\nGrid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Time taken: {grid_search_time:.2f} seconds")
    
    # Get selected features from the best pipeline
    feature_names = wine.feature_names
    k_best = best_pipeline.named_steps['feature_selection']
    selected_indices = k_best.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    print("\nSelected Features in the Best Pipeline:")
    for i, feature in enumerate(selected_features):
        print(f"{i+1}. {feature}")
    
    # Visualize the grid search results
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Analyze how feature selection affects performance
    feature_scores = []
    feature_values = param_grid['feature_selection__k']
    
    for k in feature_values:
        k_results = results[results['param_feature_selection__k'] == k]
        feature_scores.append(k_results['mean_test_score'].mean())
    
    plt.figure(figsize=(10, 6))
    plt.plot(feature_values, feature_scores, 'o-', linewidth=2)
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Mean CV Accuracy')
    plt.title('Effect of Feature Selection on Performance')
    plt.grid(True, alpha=0.3)
    plt.savefig('feature_selection_effect.png')
    plt.close()
    
    print("Feature selection effect plot saved as 'feature_selection_effect.png'")
    
    return grid_search, best_pipeline


def compare_scaler_performance():
    """
    Compare the performance of different scalers with Decision Jungles.
    """
    print("\nComparing Different Scalers with Decision Jungles")
    print("===============================================")
    
    # Load the wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Different scalers to compare
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    results = {}
    
    for scaler_name, scaler in scalers.items():
        print(f"\nEvaluating pipeline with {scaler_name}...")
        
        # Create a pipeline with the current scaler
        pipeline = Pipeline([
            ('scaler', scaler),
            ('jungle', DecisionJungleClassifier(
                n_estimators=10,
                max_width=64,
                random_state=42
            ))
        ])
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        results[scaler_name] = {
            'cv_scores': cv_scores,
            'mean_score': mean_score,
            'std_score': std_score
        }
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {mean_score:.4f} ± {std_score:.4f}")
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    
    scaler_names = list(results.keys())
    mean_scores = [results[name]['mean_score'] for name in scaler_names]
    std_scores = [results[name]['std_score'] for name in scaler_names]
    
    plt.bar(range(len(scaler_names)), mean_scores, yerr=std_scores, capsize=5)
    plt.xticks(range(len(scaler_names)), scaler_names)
    plt.ylabel('Mean CV Accuracy')
    plt.title('Decision Jungle Performance with Different Scalers')
    plt.ylim(0.9 * min(mean_scores), 1.02 * max(mean_scores))  # Adjust y-axis for better visibility
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on the bars
    for i, score in enumerate(mean_scores):
        plt.text(i, score + 0.01, f"{score:.4f}", ha='center')
    
    plt.savefig('scaler_comparison.png')
    plt.close()
    
    print("Scaler comparison plot saved as 'scaler_comparison.png'")
    
    return results


def main():
    """Main function to run the examples."""
    # Run the basic pipeline example
    basic_pipeline = basic_pipeline_example()
    
    # Run the feature selection pipeline example
    feature_pipeline = feature_selection_pipeline()
    
    # Run the PCA pipeline example
    pca_pipeline = dimensionality_reduction_pipeline()
    
    # Run the missing values pipeline example
    missing_values_pipeline = pipeline_with_missing_values()
    
    # Run the grid search pipeline example
    grid_search, best_pipeline = pipeline_grid_search()
    
    # Compare different scalers
    scaler_results = compare_scaler_performance()
    
    print("\nPipeline Integration Complete!")
    print("==============================")
    print("Key findings:")
    print("1. Decision Jungles integrate seamlessly with scikit-learn's Pipeline API")
    print("2. Feature selection can significantly reduce model size while maintaining accuracy")
    print("3. PCA can be effective for reducing dimensionality before training")
    print("4. Missing value handling is straightforward with imputation in pipelines")
    print("5. GridSearchCV allows optimizing both preprocessing steps and model parameters")
    print("6. Different scalers can have significant impact on model performance")
    
    print("\nTips for using Decision Jungles in pipelines:")
    print("- Always include scaling for best performance")
    print("- Consider feature selection to reduce memory usage")
    print("- Use pipelines for consistent preprocessing between training and inference")
    print("- Grid search over both preprocessing and model parameters for optimal results")
    print("- Test different scalers to find the best for your dataset")


if __name__ == "__main__":
    main()