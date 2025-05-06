"""
Test module for model serialization and deserialization.
"""

import os
import numpy as np
import pickle
import joblib
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_jungles import DecisionJungleClassifier


class TestSerialization:
    """Tests for model serialization and deserialization."""
    
    @pytest.fixture
    def trained_jungle(self):
        """Create a trained Decision Jungle model for testing."""
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        jungle = DecisionJungleClassifier(
            n_estimators=5,
            max_width=32,
            random_state=42
        )
        jungle.fit(X_train, y_train)
        
        return jungle, X_test, y_test
    
    @pytest.fixture
    def complex_trained_jungle(self):
        """Create a more complex trained Decision Jungle model with categorical features."""
        # Create synthetic data with categorical features
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5, 
            n_redundant=2, n_classes=3, random_state=42
        )
        
        # Make two features with strictly limited categories
        # Convert to exact category counts to ensure we stay within max_bins
        X[:, 0] = np.random.randint(0, 3, size=X.shape[0])  # 0, 1, 2 values only
        X[:, 1] = np.random.randint(0, 5, size=X.shape[0])  # 0, 1, 2, 3, 4 values only
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        jungle = DecisionJungleClassifier(
            n_estimators=5,
            max_width=32,
            max_depth=5,
            categorical_features=[0, 1],  # Explicitly specify categorical features
            max_bins=10,
            random_state=42
        )
        jungle.fit(X_train, y_train)
        
        return jungle, X_test, y_test
    
    def test_pickle_serialization(self, trained_jungle):
        """Test serialization and deserialization using pickle."""
        jungle, X_test, y_test = trained_jungle
        
        # Get original predictions
        original_preds = jungle.predict(X_test)
        original_proba = jungle.predict_proba(X_test)
        
        # Serialize and deserialize
        pickled_jungle = pickle.dumps(jungle)
        jungle_unpickled = pickle.loads(pickled_jungle)
        
        # Check if predictions are the same
        unpickled_preds = jungle_unpickled.predict(X_test)
        unpickled_proba = jungle_unpickled.predict_proba(X_test)
        
        assert np.array_equal(original_preds, unpickled_preds)
        assert np.allclose(original_proba, unpickled_proba)
        
        # Check if attributes were properly serialized
        assert jungle.n_estimators == jungle_unpickled.n_estimators
        assert jungle.get_node_count() == jungle_unpickled.get_node_count()
        assert np.array_equal(jungle.feature_importances_, jungle_unpickled.feature_importances_)
    
    def test_joblib_serialization(self, trained_jungle):
        """Test serialization and deserialization using joblib."""
        jungle, X_test, y_test = trained_jungle
        
        # Get original predictions
        original_preds = jungle.predict(X_test)
        original_proba = jungle.predict_proba(X_test)
        
        # Create temporary file
        temp_file = "temp_model.joblib"
        
        try:
            # Serialize to file
            joblib.dump(jungle, temp_file)
            
            # Deserialize from file
            jungle_loaded = joblib.load(temp_file)
            
            # Check if predictions are the same
            loaded_preds = jungle_loaded.predict(X_test)
            loaded_proba = jungle_loaded.predict_proba(X_test)
            
            assert np.array_equal(original_preds, loaded_preds)
            assert np.allclose(original_proba, loaded_proba)
            
            # Check if attributes were properly serialized
            assert jungle.n_estimators == jungle_loaded.n_estimators
            assert jungle.get_node_count() == jungle_loaded.get_node_count()
            assert np.array_equal(jungle.feature_importances_, jungle_loaded.feature_importances_)
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_complex_model_serialization(self, complex_trained_jungle):
        """Test serialization with a more complex model including categorical features."""
        jungle, X_test, y_test = complex_trained_jungle
        
        # Get original predictions
        original_preds = jungle.predict(X_test)
        
        # Serialize and deserialize
        pickled_jungle = pickle.dumps(jungle)
        jungle_unpickled = pickle.loads(pickled_jungle)
        
        # Check if predictions are the same
        unpickled_preds = jungle_unpickled.predict(X_test)
        
        assert np.array_equal(original_preds, unpickled_preds)
        
        # Check categorical features were properly serialized
        assert np.array_equal(jungle.is_categorical_, jungle_unpickled.is_categorical_)
        assert jungle.feature_bins_.keys() == jungle_unpickled.feature_bins_.keys()
        
        # Check each feature bin's values
        for feature_idx in jungle.feature_bins_:
            assert jungle.feature_bins_[feature_idx].keys() == jungle_unpickled.feature_bins_[feature_idx].keys()
            for cat in jungle.feature_bins_[feature_idx]:
                assert jungle.feature_bins_[feature_idx][cat] == jungle_unpickled.feature_bins_[feature_idx][cat]
    
    def test_parameter_changes_after_serialization(self, trained_jungle):
        """Test that parameters can be changed after serialization."""
        jungle, X_test, y_test = trained_jungle
        
        # Serialize and deserialize
        pickled_jungle = pickle.dumps(jungle)
        jungle_unpickled = pickle.loads(pickled_jungle)
        
        # Change parameters
        jungle_unpickled.set_params(n_estimators=15)
        
        # Verify parameters were changed but model still works
        assert jungle_unpickled.n_estimators == 15
        assert jungle.n_estimators != jungle_unpickled.n_estimators
        
        # Model should still be able to predict without errors
        predictions = jungle_unpickled.predict(X_test)
        assert len(predictions) == len(y_test)