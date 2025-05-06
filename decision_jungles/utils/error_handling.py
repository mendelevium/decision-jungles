"""
Error handling utilities for Decision Jungles.

This module provides custom exception classes and error handling utilities
for the Decision Jungles library to ensure consistent error reporting.
"""

from typing import Any, Dict, Optional, Union, List, Tuple, Callable, Type
import inspect
import traceback
from .logging import get_logger

# Get a logger for this module
logger = get_logger(__name__)


class DecisionJungleError(Exception):
    """Base exception for all Decision Jungle specific errors."""
    pass


class ParameterError(DecisionJungleError):
    """Exception raised for errors in the input parameters."""
    pass


class FitError(DecisionJungleError):
    """Exception raised for errors during model fitting."""
    pass


class PredictionError(DecisionJungleError):
    """Exception raised for errors during prediction."""
    pass


class NotFittedError(DecisionJungleError):
    """Exception raised when trying to use an unfitted model."""
    pass


class OptimizationError(DecisionJungleError):
    """Exception raised for errors in the optimization process."""
    pass


class ValidationError(DecisionJungleError):
    """Exception raised for validation errors."""
    pass


def check_param(param_name: str, param_value: Any, param_type: Union[Type, Tuple[Type, ...]], 
                condition: Optional[Callable[[Any], bool]] = None,
                condition_message: Optional[str] = None) -> None:
    """
    Check if a parameter meets specified type and condition requirements.
    
    Parameters
    ----------
    param_name : str
        The name of the parameter to check.
    param_value : Any
        The value of the parameter to check.
    param_type : Type or Tuple[Type, ...]
        The expected type(s) of the parameter.
    condition : Callable, optional
        A function that takes the parameter value and returns True if it meets the condition.
    condition_message : str, optional
        The error message to use if the condition fails.
        
    Raises
    ------
    ParameterError
        If the parameter does not meet the specified requirements.
    """
    # Check parameter type
    if not isinstance(param_value, param_type):
        if isinstance(param_type, tuple):
            expected_types = " or ".join([t.__name__ for t in param_type])
            raise ParameterError(f"Parameter '{param_name}' must be of type {expected_types}, got {type(param_value).__name__}.")
        else:
            raise ParameterError(f"Parameter '{param_name}' must be of type {param_type.__name__}, got {type(param_value).__name__}.")
    
    # Check additional condition if provided
    if condition is not None and not condition(param_value):
        message = condition_message or f"Parameter '{param_name}' does not meet condition requirements."
        raise ParameterError(message)


def validate_input_data(X: Any, y: Any, for_prediction: bool = False) -> Tuple[Any, Any]:
    """
    Validate input data for fit or predict methods.
    
    Parameters
    ----------
    X : array-like
        The input features.
    y : array-like or None
        The target values (None for prediction).
    for_prediction : bool, default=False
        Whether the validation is for prediction (True) or fitting (False).
        
    Returns
    -------
    X : array-like
        The validated input features.
    y : array-like or None
        The validated target values.
        
    Raises
    ------
    ValidationError
        If the input data does not meet the requirements.
    """
    from sklearn.utils.validation import check_array, check_X_y
    
    try:
        if for_prediction:
            X = check_array(X, force_all_finite='allow-nan')
            return X, y
        else:
            X, y = check_X_y(X, y, force_all_finite='allow-nan')
            return X, y
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        raise ValidationError(f"Input data validation failed: {str(e)}") from e


def log_exceptions(func: Callable) -> Callable:
    """
    Decorator to log exceptions raised by a function.
    
    Parameters
    ----------
    func : Callable
        The function to decorate.
        
    Returns
    -------
    Callable
        The decorated function.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the logger for the function's module
            logger = get_logger(func.__module__)
            
            # Log the exception with stack trace
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            
            # Re-raise the original exception
            raise
    
    # Update wrapper metadata to make it look like the original function
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    if hasattr(func, "__annotations__"):
        wrapper.__annotations__ = func.__annotations__
    
    return wrapper