"""
Decision Jungle Regressor implementation.

This module implements the main DecisionJungleRegressor class, which provides 
a scikit-learn compatible interface for Decision Jungles - an ensemble of 
directed acyclic graphs (DAGs) for regression tasks. Decision Jungles
offer reduced memory footprint and potentially improved generalization compared 
to traditional decision trees/forests through the use of node merging.

Decision Jungles were introduced in the paper:
"Decision Jungles: Compact and Rich Models for Classification" by 
Jamie Shotton, Toby Sharp, Pushmeet Kohli, Sebastian Nowozin, 
John Winn, and Antonio Criminisi (NIPS 2013).

The implementation supports different node merging algorithms (LSearch and ClusterSearch)
and various merging schedules for controlling the width of each level in the DAG.
This implementation extends the original classification-focused algorithm to regression tasks.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set, Any, Iterable, cast
import numbers
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .regression_dag import RegressionDAG


class DecisionJungleRegressor(BaseEstimator, RegressorMixin):
    """
    A decision jungle regressor implementing scikit-learn's estimator interface.
    
    Decision Jungles are ensembles of rooted decision directed acyclic graphs (DAGs)
    that offer reduced memory footprint and improved generalization compared to
    decision forests through node merging. Unlike traditional decision trees where 
    each node has exactly one parent, DAG nodes can have multiple parents, allowing
    for a more compact representation.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of DAGs in the jungle. Similar to n_estimators in RandomForest,
        increasing this value generally improves performance at the cost of training time.
    
    max_width : int, default=256
        The maximum width of each level (M parameter in the paper). This controls
        the maximum number of nodes at each depth level and is key to the memory
        efficiency of Decision Jungles.
    
    max_depth : int, default=None
        The maximum depth of the DAGs. If None, nodes are expanded until all leaves
        are pure or contain less than min_samples_split samples.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node. Values
        greater than 1 prevent overfitting.
    
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node. A split will
        only be considered if it leaves at least min_samples_leaf samples in each
        of the left and right branches.
    
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
    max_features : int, float, str, default="auto"
        The number of features to consider when looking for the best split:
        - If int, consider max_features features at each split.
        - If float, consider max_features * n_features features at each split.
        - If "auto" or "sqrt", consider sqrt(n_features) features at each split.
        - If "log2", consider log2(n_features) features at each split.
    
    criterion : str, default="mse"
        The function to measure the quality of a split. Supported criteria are:
        - "mse" for mean squared error (variance)
        - "mae" for mean absolute error
    
    random_state : int, default=None
        Controls the randomness of the estimator. The features are always randomly
        permuted at each split. This parameter also affects other randomized operations.
    
    merging_schedule : str, default="exponential"
        Controls how the width limit increases with depth:
        - "constant": Fixed width at each level (equal to max_width).
        - "exponential": Width grows exponentially (2^depth) until reaching max_width.
        - "kinect": Special schedule used in the paper for Kinect body part classification.
    
    optimization_method : str, default="lsearch"
        The algorithm used for optimizing the DAG structure:
        - "lsearch": The LSearch algorithm described in the paper (recommended).
        - "clustersearch": The ClusterSearch algorithm (alternative approach).
    
    n_jobs : int, default=None
        Number of jobs to run in parallel for both fit and predict.
        None means 1 unless in a joblib.parallel_backend context.
    
    use_optimized : bool, default=True
        Whether to use the optimized implementation of the LSearch algorithm.
        The optimized version uses NumPy vectorization for better performance.
        
    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation score is not improving.
        
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for early stopping.
        
    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.
        
    tol : float, default=1e-4
        Tolerance for the early stopping criterion.
        
    categorical_features : array-like of {bool, int, str} or None, default=None
        Indicates which features are categorical:
        - None: No features are considered categorical
        - Boolean array: Boolean mask indicating categorical features
        - Integer array: Indices of categorical features
        - String array: Names of categorical features (requires feature names during fit)
        - "auto": Automatically detect categorical features during fit
        For each categorical feature, there must be at most max_bins unique categories.
        
    max_bins : int, default=255
        Maximum number of bins to use for categorical features.
    
    Attributes
    ----------
    dags_ : list of RegressionDAG objects
        The collection of fitted DAGs (one per estimator).
    
    n_features_in_ : int
        The number of features seen during fit.
    
    feature_importances_ : ndarray of shape (n_features,)
        Importance of each feature.
    
    Examples
    --------
    >>> from decision_jungles import DecisionJungleRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    >>> reg = DecisionJungleRegressor(random_state=0)
    >>> reg.fit(X_train, y_train)
    >>> reg.predict(X_test[:3])
    array([210.52, 152.83, 127.67])
    
    Notes
    -----
    Decision Jungles were introduced by Shotton et al. in their 2013 NIPS paper.
    The key innovation is allowing nodes to have multiple parents, which enables
    significant memory savings compared to traditional decision forests.
    
    The memory efficiency comes at the cost of potentially longer training times,
    as the optimization process is more complex than standard decision tree training.
    
    This implementation extends the original algorithm to support regression tasks.
    
    References
    ----------
    .. [1] J. Shotton, T. Sharp, P. Kohli, S. Nowozin, J. Winn, and A. Criminisi,
           "Decision Jungles: Compact and Rich Models for Classification,"
           in Advances in Neural Information Processing Systems (NIPS), 2013.
    """
    
    def __init__(self, n_estimators: int = 10, max_width: int = 256,
                max_depth: Optional[int] = None, min_samples_split: int = 2,
                min_samples_leaf: int = 1, min_impurity_decrease: float = 0.0,
                max_features: Optional[Union[int, float, str]] = "auto",
                criterion: str = "mse", random_state: Optional[int] = None,
                merging_schedule: str = "exponential", optimization_method: str = "lsearch",
                n_jobs: Optional[int] = None, use_optimized: bool = True,
                early_stopping: bool = False, validation_fraction: float = 0.1,
                n_iter_no_change: int = 5, tol: float = 1e-4,
                categorical_features: Optional[Union[str, List[int], np.ndarray]] = None,
                max_bins: int = 255):
        """
        Initialize a new DecisionJungleRegressor.
        
        Args:
            n_estimators (int): Number of DAGs in the jungle.
            max_width (int): Maximum width of each level (M parameter).
            max_depth (int, optional): Maximum depth of the DAGs.
            min_samples_split (int): Minimum number of samples required to split a node.
            min_samples_leaf (int): Minimum number of samples required at a leaf node.
            min_impurity_decrease (float): Minimum impurity decrease required for a split.
            max_features (int, float, str, optional): Number of features to consider for splits.
            criterion (str): Function to measure the quality of a split ("mse" or "mae").
            random_state (int, optional): Random seed for reproducibility.
            merging_schedule (str): Type of merging schedule to use.
            optimization_method (str): Method for DAG optimization.
            n_jobs (int, optional): Number of jobs to run in parallel.
            use_optimized (bool): Whether to use the optimized implementation of the LSearch algorithm.
            early_stopping (bool): Whether to use early stopping to terminate training when validation score is not improving.
            validation_fraction (float): The proportion of training data to set aside as validation set for early stopping.
            n_iter_no_change (int): Number of iterations with no improvement to wait before early stopping.
            tol (float): Tolerance for the early stopping criterion.
            categorical_features: Indicates which features are categorical.
            max_bins (int): Maximum number of bins to use for categorical features.
        """
        self.n_estimators = n_estimators
        self.max_width = max_width
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.merging_schedule = merging_schedule
        self.optimization_method = optimization_method
        self.n_jobs = n_jobs
        self.use_optimized = use_optimized
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.categorical_features = categorical_features
        self.max_bins = max_bins
        
        # Validate parameters immediately during initialization
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """
        Validate the parameters passed to the constructor.
        
        Raises:
            ValueError: If any parameters are invalid.
        """
        if not isinstance(self.n_estimators, numbers.Integral) or self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be a positive integer, got {self.n_estimators}")
        
        if not isinstance(self.max_width, numbers.Integral) or self.max_width <= 0:
            raise ValueError(f"max_width must be a positive integer, got {self.max_width}")
        
        if self.max_depth is not None:
            if not isinstance(self.max_depth, numbers.Integral) or self.max_depth <= 0:
                raise ValueError(f"max_depth must be a positive integer or None, got {self.max_depth}")
        
        if not isinstance(self.min_samples_split, numbers.Integral) or self.min_samples_split <= 1:
            raise ValueError(f"min_samples_split must be an integer > 1, got {self.min_samples_split}")
        
        if not isinstance(self.min_samples_leaf, numbers.Integral) or self.min_samples_leaf <= 0:
            raise ValueError(f"min_samples_leaf must be a positive integer, got {self.min_samples_leaf}")
        
        if not isinstance(self.min_impurity_decrease, numbers.Real) or self.min_impurity_decrease < 0:
            raise ValueError(f"min_impurity_decrease must be a non-negative float, got {self.min_impurity_decrease}")
        
        if isinstance(self.max_features, str):
            if self.max_features not in ("auto", "sqrt", "log2"):
                raise ValueError(f"max_features must be 'auto', 'sqrt', or 'log2', got {self.max_features}")
        elif isinstance(self.max_features, numbers.Integral):
            if self.max_features <= 0:
                raise ValueError(f"max_features must be a positive integer, got {self.max_features}")
        elif isinstance(self.max_features, numbers.Real):
            if self.max_features <= 0 or self.max_features > 1:
                raise ValueError(f"max_features must be in (0, 1], got {self.max_features}")
        elif self.max_features is not None:
            raise ValueError(f"max_features must be an int, float, or one of {{'auto', 'sqrt', 'log2'}}, got {self.max_features}")
        
        if self.criterion not in ("mse", "mae"):
            raise ValueError(f"criterion must be 'mse' or 'mae', got {self.criterion}")
        
        if self.merging_schedule not in ("constant", "exponential", "kinect"):
            raise ValueError(f"merging_schedule must be 'constant', 'exponential', or 'kinect', got {self.merging_schedule}")
        
        if self.optimization_method not in ("lsearch", "clustersearch"):
            raise ValueError(f"optimization_method must be 'lsearch' or 'clustersearch', got {self.optimization_method}")
        
        # Validate early stopping parameters
        if not isinstance(self.early_stopping, bool):
            raise ValueError(f"early_stopping must be a boolean, got {self.early_stopping}")
        
        if not isinstance(self.validation_fraction, numbers.Real) or self.validation_fraction <= 0 or self.validation_fraction >= 1:
            raise ValueError(f"validation_fraction must be a float in (0, 1), got {self.validation_fraction}")
        
        if not isinstance(self.n_iter_no_change, numbers.Integral) or self.n_iter_no_change <= 0:
            raise ValueError(f"n_iter_no_change must be a positive integer, got {self.n_iter_no_change}")
        
        if not isinstance(self.tol, numbers.Real) or self.tol < 0:
            raise ValueError(f"tol must be a non-negative float, got {self.tol}")
            
        # Validate categorical feature parameters
        if self.categorical_features is not None and not isinstance(self.categorical_features, str) and not hasattr(self.categorical_features, '__len__'):
            raise ValueError(f"categorical_features must be None, 'auto', or an array-like of bool, int, str, got {self.categorical_features}")
        
        if self.categorical_features is not None and isinstance(self.categorical_features, str) and self.categorical_features != "auto":
            raise ValueError(f"If categorical_features is a string, it must be 'auto', got {self.categorical_features}")
        
        if not isinstance(self.max_bins, numbers.Integral) or self.max_bins <= 1:
            raise ValueError(f"max_bins must be an integer > 1, got {self.max_bins}")
    
    def _fit_dag(self, dag: RegressionDAG, X: np.ndarray, y: np.ndarray) -> RegressionDAG:
        """
        Fit a single DAG to the data.
        
        Args:
            dag (RegressionDAG): The DAG to fit.
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training target values.
            
        Returns:
            RegressionDAG: The fitted DAG.
        """
        dag.fit(X, y)
        return dag
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionJungleRegressor':
        """
        Build a decision jungle ensemble from the training data.
        
        The method constructs an ensemble of DAGs (Directed Acyclic Graphs), where each
        DAG is trained on the complete dataset but with randomized feature selection.
        This is similar to how Random Forests work, but with the additional step of
        optimizing the DAG structure through node merging.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
            
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : DecisionJungleRegressor
            Fitted estimator.
            
        Notes
        -----
        The training process constructs an ensemble of DAGs level by level. At each level:
        1. A maximum width constraint is enforced based on the merging schedule
        2. The optimization algorithm is used to determine:
           - Optimal split parameters for each parent node
           - Optimal branch assignments to child nodes
        3. This process creates a more compact model by allowing multiple parent nodes
           to share the same child nodes.
        
        If early_stopping is enabled, a portion of the training data is used as a
        validation set. Training stops when validation performance stops improving
        for n_iter_no_change iterations within the specified tolerance.
        """
        # Validate parameters
        self._validate_parameters()
        
        # Check input data
        X, y = check_X_y(X, y, force_all_finite='allow-nan', y_numeric=True)
        
        # Process categorical features
        self._process_categorical_features(X)
        
        # Set up validation data for early stopping if enabled
        validation_X, validation_y = None, None
        training_X, training_y = X, y
        
        if self.early_stopping:
            # Split out a validation set
            rng = np.random.RandomState(self.random_state)
            indices = np.arange(X.shape[0])
            rng.shuffle(indices)
            
            # Calculate split point
            val_size = int(X.shape[0] * self.validation_fraction)
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            # Create validation set
            validation_X = X[val_indices]
            validation_y = y[val_indices]
            
            # Use remaining samples for training
            training_X = X[train_indices]
            training_y = y[train_indices]
            
            # Track if early stopping was triggered
            self.stopped_early_ = False
        
        # Create DAGs
        self.dags_ = []
        random_states = None
        
        if self.random_state is not None:
            # Generate different random states for each DAG
            rng = np.random.RandomState(self.random_state)
            random_states = rng.randint(np.iinfo(np.int32).max, size=self.n_estimators)
        
        # Define a function to create a DAG with the given parameters
        def create_dag(i):
            rs = None if random_states is None else random_states[i]
            return RegressionDAG(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                criterion=self.criterion,
                random_state=rs,
                merging_schedule=self.merging_schedule,
                max_width=self.max_width,
                optimization_method=self.optimization_method,
                use_optimized=self.use_optimized,
                early_stopping=self.early_stopping,
                validation_X=validation_X,
                validation_y=validation_y,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                is_categorical=self.is_categorical_,
                feature_bins=self.feature_bins_
            )
        
        # Fit DAGs in parallel
        if self.n_jobs is None or self.n_jobs == 1:
            # Sequential training
            for i in range(self.n_estimators):
                dag = create_dag(i)
                self._fit_dag(dag, training_X, training_y)
                self.dags_.append(dag)
                
                # Update early stopping flag if any DAG stopped early
                if self.early_stopping and dag.stopped_early:
                    self.stopped_early_ = True
        else:
            # Parallel training
            dags = [create_dag(i) for i in range(self.n_estimators)]
            self.dags_ = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._fit_dag)(dag, training_X, training_y)
                for dag in dags
            )
            
            # Update early stopping flag if any DAG stopped early
            if self.early_stopping and any(dag.stopped_early for dag in self.dags_):
                self.stopped_early_ = True
        
        # Store feature and sample dimensions
        self.n_features_in_ = X.shape[1]
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for the input samples.
        
        The prediction of an input sample is the average of the predictions of
        all DAGs in the ensemble.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
            
        Notes
        -----
        In Decision Jungles, prediction involves routing each sample through each DAG
        in the ensemble. Because DAGs can have multiple paths to the same leaf node,
        the prediction process can be more complex than in traditional decision trees,
        but the final prediction calculation is similar (averaging across all
        DAGs in the ensemble).
        """
        # Check if fit has been called
        check_is_fitted(self, ["dags_"])
        
        # Input validation - allow NaN values explicitly
        X = check_array(X, force_all_finite='allow-nan')
        
        # Check if the input feature dimensionality is consistent with the training data
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but DecisionJungleRegressor is expecting {self.n_features_in_} features.")
        
        # Get predictions from each DAG
        dag_preds = np.array([dag.predict(X) for dag in self.dags_])
        
        # Average predictions across DAGs
        return np.mean(dag_preds, axis=0)
    
    def get_memory_usage(self) -> int:
        """
        Calculate the total memory usage of the jungle in bytes.
        
        This method estimates the memory usage by summing the memory consumption
        of all DAGs in the ensemble. Each DAG reports its own memory usage based
        on the number and types of nodes it contains.
        
        Returns
        -------
        int
            Total memory usage in bytes.
            
        Notes
        -----
        This is one of the key metrics for evaluating Decision Jungles, as their
        primary advantage over Decision Forests is reduced memory consumption while
        maintaining similar or better predictive performance.
        
        Memory usage is calculated by counting the storage requirements for split
        nodes (feature indices, thresholds, and child pointers) and leaf nodes
        (target value arrays).
        """
        check_is_fitted(self, ["dags_"])
        
        return sum(dag.get_memory_usage() for dag in self.dags_)
    
    def get_node_count(self) -> int:
        """
        Get the total number of nodes in the jungle.
        
        This method counts the total number of nodes (both split nodes and leaf nodes)
        across all DAGs in the ensemble. This is a key metric for comparing the model
        size of Decision Jungles versus traditional Decision Forests.
        
        Returns
        -------
        int
            Total number of nodes across all DAGs in the ensemble.
            
        Notes
        -----
        Due to the node merging process in Decision Jungles, the total node count
        is typically significantly lower than an equivalent Decision Forest with
        the same number of estimators and similar predictive performance.
        
        This reduced node count directly translates to memory savings, which is
        one of the primary advantages of Decision Jungles.
        """
        check_is_fitted(self, ["dags_"])
        
        return sum(dag.get_node_count() for dag in self.dags_)
    
    def get_max_depth(self) -> int:
        """
        Get the maximum depth of all DAGs in the jungle.
        
        This method returns the maximum depth across all DAGs in the ensemble.
        The depth of a DAG is the length of the longest path from the root node
        to any leaf node.
        
        Returns
        -------
        int
            Maximum depth across all DAGs in the ensemble.
            
        Notes
        -----
        The maximum depth influences both the model's complexity and its ability
        to capture intricate patterns in the data. A deeper model can potentially
        represent more complex decision boundaries, but may also be more prone to
        overfitting if not properly regularized.
        
        In Decision Jungles, the merging of nodes can lead to more efficient use
        of the model's depth compared to traditional Decision Trees, as multiple
        paths can lead to shared nodes.
        """
        check_is_fitted(self, ["dags_"])
        
        return max(dag.get_max_depth() for dag in self.dags_)
        
    def _process_categorical_features(self, X: np.ndarray) -> None:
        """
        Process categorical features.
        
        This method identifies categorical features based on the categorical_features
        parameter and prepares them for use in the model. It creates a boolean mask
        (is_categorical_) to indicate which features are categorical.
        
        Args:
            X: Input features
        """
        n_features = X.shape[1]
        
        # Default: no categorical features
        self.is_categorical_: np.ndarray = np.zeros(n_features, dtype=bool)
        self.feature_bins_: Dict[int, Dict[Any, int]] = {}
        
        if self.categorical_features is None:
            return
            
        # Process categorical features
        if self.categorical_features == "auto":
            # Try to detect categorical features automatically
            # In pure numpy arrays, we'll consider integer types as potential categorical
            for i in range(n_features):
                col = X[:, i]
                n_unique = len(np.unique(col[~np.isnan(col)]))
                # If we have a small number of unique values, treat it as categorical
                if n_unique <= self.max_bins and n_unique > 1:
                    self.is_categorical_[i] = True
                    # Extract unique categories and assign bin indices
                    categories = np.unique(col[~np.isnan(col)])
                    self.feature_bins_[i] = {cat: idx for idx, cat in enumerate(categories)}
        else:
            # Process explicitly specified categorical features
            categorical_indices = []
            
            if isinstance(self.categorical_features[0], bool):
                # Boolean mask
                if len(self.categorical_features) != n_features:
                    raise ValueError(f"categorical_features as a boolean mask must have length n_features={n_features},"
                                    f" got {len(self.categorical_features)}")
                self.is_categorical_ = np.asarray(self.categorical_features, dtype=bool)
                categorical_indices = np.where(self.is_categorical_)[0]
            elif isinstance(self.categorical_features[0], (int, np.integer)):
                # Integer indices
                categorical_indices = self.categorical_features
                for idx in categorical_indices:
                    if not (0 <= idx < n_features):
                        raise ValueError(f"categorical_features as indices must be in [0, {n_features-1}],"
                                        f" got {idx}")
                self.is_categorical_[categorical_indices] = True
            else:
                # Only other option is string feature names, which we don't handle without feature names
                raise ValueError("String categorical_features require DataFrame input with feature names")
                
            # Process selected categorical features
            for i in categorical_indices:
                col = X[:, i]
                categories = np.unique(col[~np.isnan(col)])
                if len(categories) > self.max_bins:
                    raise ValueError(f"Categorical feature at index {i} has {len(categories)} unique "
                                   f"values, which exceeds max_bins={self.max_bins}")
                # Create category-to-bin mapping
                self.feature_bins_[i] = {cat: idx for idx, cat in enumerate(categories)}
    
    def _calculate_feature_importances(self) -> None:
        """
        Calculate feature importances for the forest.
        
        Feature importances are calculated based on how frequently features are used
        in split nodes across all DAGs in the ensemble. This method sets the
        feature_importances_ attribute.
        
        The importance of a feature is calculated as the normalized count of how many
        times that feature is used for splitting across all nodes in all DAGs.
        
        Notes
        -----
        This implementation differs from traditional tree-based feature importance, which
        often incorporates impurity reduction. The current approach
        focuses on feature usage frequency, which is a simpler but still informative
        measure for Decision Jungles.
        """
        check_is_fitted(self, ["dags_", "n_features_in_"])
        
        # Initialize feature counts
        feature_counts = np.zeros(self.n_features_in_)
        
        # Count feature usage across all DAGs
        for dag in self.dags_:
            for node_id in dag.nodes:
                node = dag.nodes[node_id]
                if hasattr(node, 'feature_idx'):  # Split node
                    feature_counts[node.feature_idx] += 1
        
        # Calculate normalized importance
        total_splits = np.sum(feature_counts)
        if total_splits > 0:
            self._feature_importances = feature_counts / total_splits
        else:
            self._feature_importances = np.zeros_like(feature_counts)
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """
        The feature importances based on feature usage in the jungle.
        
        The importance of a feature is calculated as the normalized count of how many
        times that feature is used for splitting across all nodes in all DAGs of the
        ensemble.
        
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized array of feature importances. Higher values indicate more 
            important features.
            
        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        check_is_fitted(self, ["_feature_importances"])
        return self._feature_importances
        
    def __getstate__(self) -> Dict[str, Any]:
        """
        Return the state of the estimator for pickling.
        
        This method is called when pickling the estimator. It returns the internal state
        of the object, which can then be used to restore the object when unpickling.
        
        Returns
        -------
        state : dict
            The state of the estimator to be serialized.
        """
        return self.__dict__.copy()
        
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Set the state of the estimator after unpickling.
        
        This method is called when unpickling the estimator. It restores the internal
        state of the object.
        
        Parameters
        ----------
        state : dict
            The state of the estimator as returned by __getstate__().
        """
        self.__dict__.update(state)