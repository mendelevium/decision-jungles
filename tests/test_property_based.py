"""
Property-based tests for Decision Jungles using Hypothesis.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from decision_jungles import DecisionJungleClassifier
from decision_jungles.dag import DAG


# Define the hypothesis search strategies
# Create arrays with reasonable dimensionality
features_arrays = arrays(
    dtype=np.float64,
    shape=st.tuples(
        st.integers(min_value=5, max_value=50),  # n_samples
        st.integers(min_value=2, max_value=20),  # n_features
    ),
    elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)

labels_strategy = st.lists(
    st.integers(min_value=0, max_value=5),  # limit to 6 classes (0-5)
    min_size=5,
    max_size=50,
)


@pytest.mark.skip(reason="Hypothesis tests are time-consuming, run explicitly when needed")
@settings(max_examples=10, deadline=None)
@given(X=features_arrays, y_raw=labels_strategy)
def test_model_consistency(X, y_raw):
    """Test that the model behaves consistently."""
    # Ensure y has the correct length
    y = np.array(y_raw[:X.shape[0]])
    
    # Ensure we have at least 2 classes
    if len(np.unique(y)) < 2:
        y[0] = 0
        y[1] = 1
    
    # First model
    clf1 = DecisionJungleClassifier(
        n_estimators=3,
        max_width=4,
        max_depth=3,
        random_state=42
    )
    clf1.fit(X, y)
    pred1 = clf1.predict(X)
    
    # Second model with identical parameters
    clf2 = DecisionJungleClassifier(
        n_estimators=3,
        max_width=4,
        max_depth=3,
        random_state=42
    )
    clf2.fit(X, y)
    pred2 = clf2.predict(X)
    
    # Different random_state should give different results
    clf3 = DecisionJungleClassifier(
        n_estimators=3,
        max_width=4,
        max_depth=3,
        random_state=43
    )
    clf3.fit(X, y)
    pred3 = clf3.predict(X)
    
    # Property 1: Identical models with same random_state should give identical predictions
    assert np.array_equal(pred1, pred2)
    
    # Property 2: Different parameters should generally give different predictions
    # Note: This property is probabilistic and might fail occasionally
    if len(X) > 10 and len(np.unique(y)) >= 2:
        different_preds = ~np.array_equal(pred1, pred3)
        if not different_preds:
            # It's okay if they're equal occasionally, but we should note it
            print("Warning: Different random states produced identical predictions")


@pytest.mark.skip(reason="Hypothesis tests are time-consuming, run explicitly when needed")
@settings(max_examples=10, deadline=None)
@given(X=features_arrays, y_raw=labels_strategy)
def test_probability_properties(X, y_raw):
    """Test properties related to probability predictions."""
    # Ensure y has the correct length
    y = np.array(y_raw[:X.shape[0]])
    
    # Ensure we have at least 2 classes
    if len(np.unique(y)) < 2:
        y[0] = 0
        y[1] = 1
    
    # Fit the model
    clf = DecisionJungleClassifier(
        n_estimators=3,
        max_width=4,
        max_depth=3,
        random_state=42
    )
    clf.fit(X, y)
    
    # Get predictions and probabilities
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    
    # Property 1: Probabilities should sum to 1 for each sample
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    
    # Property 2: All probabilities should be between 0 and 1
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
    
    # Property 3: The class with highest probability should match the predicted class
    pred_from_proba = np.argmax(probs, axis=1)
    assert np.array_equal(preds, pred_from_proba)


@pytest.mark.skip(reason="Hypothesis tests are time-consuming, run explicitly when needed")
@settings(max_examples=10, deadline=None)
@given(X=features_arrays, y_raw=labels_strategy)
def test_parameter_effects(X, y_raw):
    """Test that increasing certain parameters has the expected effects."""
    # Ensure y has the correct length
    y = np.array(y_raw[:X.shape[0]])
    
    # Ensure we have at least 2 classes
    if len(np.unique(y)) < 2:
        y[0] = 0
        y[1] = 1
    
    # Models with different parameters
    clf1 = DecisionJungleClassifier(
        n_estimators=3,
        max_width=4,
        max_depth=2,
        random_state=42
    )
    
    clf2 = DecisionJungleClassifier(
        n_estimators=5,  # More estimators
        max_width=4,
        max_depth=2,
        random_state=42
    )
    
    # Fit models
    clf1.fit(X, y)
    clf2.fit(X, y)
    
    # Property 1: More estimators should use more memory
    assert clf2.get_memory_usage() >= clf1.get_memory_usage()
    
    # Property 2: More estimators should have more total nodes
    assert clf2.get_node_count() >= clf1.get_node_count()