"""
Example showing the usage of categorical features in Decision Jungles.

This example demonstrates how Decision Jungles can handle categorical features
directly, without the need for one-hot encoding or other preprocessing steps.
It compares performance with and without categorical feature handling on datasets
with categorical features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_jungles import DecisionJungleClassifier
from sklearn.ensemble import RandomForestClassifier


# Helper functions for visualization
def plot_comparison(results, title="Model Comparison"):
    """Plot the comparison of different models and configurations."""
    models = list(results.keys())
    metrics = list(results[models[0]].keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        model_names = []
        values = []
        
        for model in models:
            model_names.append(model)
            values.append(results[model][metric])
        
        axes[i].bar(model_names, values)
        axes[i].set_title(f"{metric}")
        axes[i].set_ylim(0, 1.05)
        
        # Add value labels
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha='center')
            
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    return fig


def load_car_evaluation_dataset():
    """Load and prepare the Car Evaluation dataset which has categorical features."""
    # Load the car evaluation dataset
    car = fetch_openml(name='car', version=3, as_frame=True)
    X = car.data
    y = car.target
    
    # Map class labels to integers (unacc=0, acc=1, good=2, vgood=3)
    class_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    y = y.map(class_mapping)
    
    # For sklearn models, we need to map categorical features to integers
    # For Decision Jungles with categorical_features="auto", the strings would
    # also need to be converted to numeric values first
    
    # Map categorical features to integers
    for col in X.columns:
        X[col] = X[col].astype('category').cat.codes
        
    return X, y


def load_adult_income_dataset():
    """Load and prepare the Adult Income dataset (also known as Census Income)."""
    # Load the adult income dataset
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    X = adult.data
    y = adult.target
    
    # Map class labels to integers (<=50K=0, >50K=1)
    y = (y == ">50K").astype(int)
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Convert categorical columns to integer codes
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
    
    return X, y, cat_cols


def main():
    # Example 1: Car Evaluation Dataset
    print("\n=== Car Evaluation Dataset ===")
    X, y = load_car_evaluation_dataset()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 1. Decision Jungle with automatic categorical feature detection
    print("\nTraining Decision Jungle with categorical features...")
    dj_cat = DecisionJungleClassifier(
        n_estimators=50,
        categorical_features="auto",
        max_bins=20,
        random_state=42
    )
    dj_cat.fit(X_train, y_train)
    
    # 2. Standard Decision Jungle (treating all features as continuous)
    print("Training standard Decision Jungle...")
    dj_std = DecisionJungleClassifier(
        n_estimators=50,
        random_state=42
    )
    dj_std.fit(X_train, y_train)
    
    # 3. Random Forest (for comparison)
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=50, 
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Evaluate all models
    results = {}
    
    models = {
        "DJ (categorical)": dj_cat,
        "DJ (standard)": dj_std,
        "Random Forest": rf
    }
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        results[name] = {
            "Accuracy": accuracy,
            "F1 (macro)": f1_macro
        }
        
        print(f"\n{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score (macro): {f1_macro:.4f}")
        
        # For Decision Jungles, also report memory usage
        if 'DJ' in name:
            mem_usage = model.get_memory_usage() / 1024  # Convert to KB
            node_count = model.get_node_count()
            print(f"  Memory Usage: {mem_usage:.2f} KB")
            print(f"  Total Nodes: {node_count}")
    
    # Plot the results
    fig = plot_comparison(results, title="Car Evaluation Dataset Results")
    plt.savefig('car_evaluation_results.png')
    
    # Example 2: Adult Income Dataset (more complex)
    print("\n\n=== Adult Income Dataset ===")
    X, y, cat_cols = load_adult_income_dataset()
    
    # Identify categorical column indices
    cat_indices = [X.columns.get_loc(col) for col in cat_cols]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 1. Decision Jungle with explicit categorical features
    print("\nTraining Decision Jungle with categorical features...")
    dj_cat = DecisionJungleClassifier(
        n_estimators=50,
        categorical_features=cat_indices,
        max_bins=20,
        random_state=42
    )
    dj_cat.fit(X_train, y_train)
    
    # 2. Standard Decision Jungle (treating all features as continuous)
    print("Training standard Decision Jungle...")
    dj_std = DecisionJungleClassifier(
        n_estimators=50,
        random_state=42
    )
    dj_std.fit(X_train, y_train)
    
    # 3. Decision Jungle with one-hot encoded categorical features
    print("Training Decision Jungle with one-hot encoding...")
    # Create a column transformer for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_indices)
        ],
        remainder='passthrough'
    )
    
    # Create a pipeline with preprocessing and model
    dj_ohe_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionJungleClassifier(n_estimators=50, random_state=42))
    ])
    
    dj_ohe_pipeline.fit(X_train, y_train)
    
    # 4. Random Forest (for comparison)
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=50, 
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Evaluate all models
    results = {}
    
    models = {
        "DJ (categorical)": dj_cat,
        "DJ (standard)": dj_std,
        "DJ (one-hot)": dj_ohe_pipeline,
        "Random Forest": rf
    }
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_binary = f1_score(y_test, y_pred, average='binary')
        
        results[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1_binary
        }
        
        print(f"\n{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1_binary:.4f}")
        
        # For Decision Jungles, also report memory usage
        if 'DJ' in name and 'one-hot' not in name:
            mem_usage = model.get_memory_usage() / 1024  # Convert to KB
            node_count = model.get_node_count()
            print(f"  Memory Usage: {mem_usage:.2f} KB")
            print(f"  Total Nodes: {node_count}")
    
    # Plot the results
    fig = plot_comparison(results, title="Adult Income Dataset Results")
    plt.savefig('adult_income_results.png')
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()