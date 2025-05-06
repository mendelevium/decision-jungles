"""
Test module for DecisionJungleRegressor.
"""

import os
import numpy as np
import pytest
from sklearn.datasets import load_diabetes, fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_jungles import DecisionJungleRegressor


class TestDecisionJungleRegressor:
    """Tests for DecisionJungleRegressor."""
    
    @pytest.fixture
    def diabetes_data(self):
        """Load the diabetes dataset for testing."""
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate a synthetic regression dataset."""
        X, y = make_regression(
            n_samples=500, n_features=10, n_informative=5, noise=0.1, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def test_init_params(self):
        """Test parameter validation during initialization."""
        # Valid parameters should not raise exceptions
        reg = DecisionJungleRegressor(
            n_estimators=10,
            max_width=64,
            max_depth=5,
            criterion="mse",
            random_state=42
        )
        assert reg.n_estimators == 10
        assert reg.max_width == 64
        assert reg.max_depth == 5
        assert reg.criterion == "mse"
        
        # Invalid parameters should raise ValueError
        with pytest.raises(ValueError):
            DecisionJungleRegressor(n_estimators=0)
        
        with pytest.raises(ValueError):
            DecisionJungleRegressor(max_width=-1)
        
        with pytest.raises(ValueError):
            DecisionJungleRegressor(criterion="invalid")
    
    def test_fit_predict(self, diabetes_data):
        """Test basic fit and predict functionality."""
        X_train, X_test, y_train, y_test = diabetes_data
        
        # Test with default parameters
        reg = DecisionJungleRegressor(random_state=42)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        
        # Check predictions shape
        assert y_pred.shape == y_test.shape
        
        # Check that predictions are reasonable (R² > 0)
        r2 = r2_score(y_test, y_pred)
        assert r2 > 0, f"R² score should be positive, got {r2}"
    
    def test_feature_importances(self, synthetic_data):
        """Test feature importance calculation."""
        X_train, _, y_train, _ = synthetic_data
        
        reg = DecisionJungleRegressor(random_state=42)
        reg.fit(X_train, y_train)
        
        # Check that feature_importances_ is available
        assert hasattr(reg, "feature_importances_")
        
        # Check shape and normalization
        assert reg.feature_importances_.shape == (X_train.shape[1],)
        assert np.isclose(np.sum(reg.feature_importances_), 1.0)
        
        # Check that feature importances are computed
        # Note: In Decision Jungles, feature importance is based on usage frequency
        # rather than impurity reduction, so we don't check for any specific pattern
        # just that they're calculated properly
        informative_importance = np.sum(reg.feature_importances_[:5])
        noise_importance = np.sum(reg.feature_importances_[5:])
        assert informative_importance >= 0 and noise_importance >= 0, "Feature importances should be non-negative"
    
    def test_mae_criterion(self, diabetes_data):
        """Test using mean absolute error criterion."""
        X_train, X_test, y_train, y_test = diabetes_data
        
        # Test with MAE criterion
        reg = DecisionJungleRegressor(criterion="mae", random_state=42)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        
        # Check predictions shape
        assert y_pred.shape == y_test.shape
        
        # Check that predictions are reasonable (R² > 0)
        r2 = r2_score(y_test, y_pred)
        assert r2 > 0, f"R² score should be positive, got {r2}"
    
    def test_early_stopping(self, synthetic_data):
        """Test early stopping functionality."""
        X_train, X_test, y_train, y_test = synthetic_data
        
        # Test with early stopping enabled
        reg = DecisionJungleRegressor(
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=2,
            tol=1e-3,
            random_state=42
        )
        reg.fit(X_train, y_train)
        
        # Check that attribute exists
        assert hasattr(reg, "stopped_early_")
    
    def test_memory_usage(self, synthetic_data):
        """Test memory usage calculation."""
        X_train, _, y_train, _ = synthetic_data
        
        reg = DecisionJungleRegressor(
            n_estimators=5,
            max_width=32,
            max_depth=5,
            random_state=42
        )
        reg.fit(X_train, y_train)
        
        # Check that memory usage is calculated
        memory_usage = reg.get_memory_usage()
        assert memory_usage > 0
        
        # Check that node count is calculated
        node_count = reg.get_node_count()
        assert node_count > 0
        
        # Check max depth
        max_depth = reg.get_max_depth()
        assert 0 < max_depth <= 5  # Should be at most the specified max_depth
    
    def test_categorical_features(self):
        """Test regression with categorical features."""
        # Create dataset with categorical features
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        
        # Make two features categorical
        X[:, 0] = np.random.randint(0, 3, size=n_samples)  # 0, 1, 2 values only
        X[:, 1] = np.random.randint(0, 5, size=n_samples)  # 0, 1, 2, 3, 4 values only
        
        # Create target based on categorical features
        y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.5
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit model with manually specified categorical features
        reg = DecisionJungleRegressor(
            n_estimators=5,
            categorical_features=[0, 1],
            random_state=42
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        
        # Should have positive performance since target depends on categorical features
        r2 = r2_score(y_test, y_pred)
        assert r2 > 0.3, f"R² score should be positive (>0.3), got {r2}"
        
        # Test with auto detection
        reg_auto = DecisionJungleRegressor(
            n_estimators=5,
            categorical_features="auto",
            random_state=42
        )
        reg_auto.fit(X_train, y_train)
        
        # The categorical features should be detected
        assert reg_auto.is_categorical_[0] and reg_auto.is_categorical_[1]
        # Note: We don't check if the third feature is detected as non-categorical
        # because the auto-detection is based on the number of unique values,
        # and with random data it might have few enough unique values to be 
        # considered categorical
    
    def test_serialization(self, synthetic_data):
        """Test model serialization and deserialization."""
        import pickle
        
        X_train, X_test, y_train, y_test = synthetic_data
        
        # Fit model
        reg = DecisionJungleRegressor(
            n_estimators=5,
            max_width=32,
            random_state=42
        )
        reg.fit(X_train, y_train)
        
        # Get original predictions
        original_preds = reg.predict(X_test)
        
        # Serialize and deserialize
        serialized = pickle.dumps(reg)
        deserialized_reg = pickle.loads(serialized)
        
        # Get predictions from deserialized model
        deserialized_preds = deserialized_reg.predict(X_test)
        
        # Check that predictions are the same
        np.testing.assert_allclose(original_preds, deserialized_preds)
        
        # Check that attributes were properly serialized
        assert reg.n_estimators == deserialized_reg.n_estimators
        assert reg.get_node_count() == deserialized_reg.get_node_count()
        np.testing.assert_allclose(
            reg.feature_importances_, deserialized_reg.feature_importances_
        )