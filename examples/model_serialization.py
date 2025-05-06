"""
Example demonstrating model serialization and deserialization.

This example shows how to save and load Decision Jungle models using both
pickle and joblib serialization methods.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_jungles import DecisionJungleClassifier


def time_prediction(model, X, n_repeats=10):
    """Measure prediction time."""
    times = []
    for _ in range(n_repeats):
        start = time.time()
        model.predict(X)
        end = time.time()
        times.append(end - start)
    return np.mean(times), np.std(times)


def main():
    # Load datasets
    print("Loading datasets...")
    datasets = {
        'iris': load_iris(return_X_y=True),
        'breast_cancer': load_breast_cancer(return_X_y=True)
    }
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n=== Dataset: {dataset_name} ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train a Decision Jungle model
        print("Training model...")
        jungle = DecisionJungleClassifier(
            n_estimators=10,
            max_width=64,
            random_state=42
        )
        jungle.fit(X_train, y_train)
        
        # Make predictions with the original model
        y_pred_original = jungle.predict(X_test)
        accuracy_original = accuracy_score(y_test, y_pred_original)
        orig_time, _ = time_prediction(jungle, X_test)
        print(f"Original model accuracy: {accuracy_original:.4f}")
        print(f"Original model prediction time: {orig_time:.6f} seconds")
        
        # Print some model information
        print(f"Model parameters: n_estimators={jungle.n_estimators}, max_width={jungle.max_width}")
        print(f"Number of DAGs: {len(jungle.dags_)}")
        print(f"Total nodes: {jungle.get_node_count()}")
        print(f"Memory usage: {jungle.get_memory_usage() / 1024:.2f} KB")
        
        # 1. Serialize/deserialize using pickle
        print("\nSerializing with pickle...")
        
        # Serialize
        pickle_file = f"{dataset_name}_jungle.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(jungle, f)
        
        pickle_size = os.path.getsize(pickle_file) / 1024
        print(f"Serialized model size: {pickle_size:.2f} KB")
        
        # Deserialize
        print("Deserializing with pickle...")
        with open(pickle_file, 'rb') as f:
            jungle_pickle = pickle.load(f)
        
        # Make predictions with the deserialized model
        y_pred_pickle = jungle_pickle.predict(X_test)
        accuracy_pickle = accuracy_score(y_test, y_pred_pickle)
        pickle_time, _ = time_prediction(jungle_pickle, X_test)
        
        print(f"Deserialized model (pickle) accuracy: {accuracy_pickle:.4f}")
        print(f"Prediction time: {pickle_time:.6f} seconds")
        print(f"Predictions match original: {np.array_equal(y_pred_original, y_pred_pickle)}")
        
        # 2. Serialize/deserialize using joblib
        print("\nSerializing with joblib...")
        
        # Serialize
        joblib_file = f"{dataset_name}_jungle.joblib"
        joblib.dump(jungle, joblib_file)
        
        joblib_size = os.path.getsize(joblib_file) / 1024
        print(f"Serialized model size: {joblib_size:.2f} KB")
        
        # Deserialize
        print("Deserializing with joblib...")
        jungle_joblib = joblib.load(joblib_file)
        
        # Make predictions with the deserialized model
        y_pred_joblib = jungle_joblib.predict(X_test)
        accuracy_joblib = accuracy_score(y_test, y_pred_joblib)
        joblib_time, _ = time_prediction(jungle_joblib, X_test)
        
        print(f"Deserialized model (joblib) accuracy: {accuracy_joblib:.4f}")
        print(f"Prediction time: {joblib_time:.6f} seconds")
        print(f"Predictions match original: {np.array_equal(y_pred_original, y_pred_joblib)}")
        
        # Compare feature importances
        print("\nFeature importance comparison:")
        print(f"Original vs Pickle correlation: {np.corrcoef(jungle.feature_importances_, jungle_pickle.feature_importances_)[0, 1]:.6f}")
        print(f"Original vs Joblib correlation: {np.corrcoef(jungle.feature_importances_, jungle_joblib.feature_importances_)[0, 1]:.6f}")
        
        # Cleanup
        print("\nCleaning up files...")
        os.remove(pickle_file)
        os.remove(joblib_file)
        
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()