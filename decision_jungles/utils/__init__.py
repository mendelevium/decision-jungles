"""
Utility functions for Decision Jungles.
"""

from .metrics import memory_usage
from .visualization import plot_dag
from .logging import (
    setup_logging,
    get_logger,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
)
from .error_handling import (
    DecisionJungleError,
    ParameterError,
    FitError,
    PredictionError,
    NotFittedError,
    OptimizationError,
    ValidationError,
    check_param,
    validate_input_data,
    log_exceptions
)

# Import memory profiling utilities if dependencies are available
try:
    from .memory_profiling import (
        measure_model_memory,
        memory_usage_vs_accuracy,
        estimate_model_size
    )
except ImportError:
    # Optional dependencies may not be available
    pass

# Import benchmarking utilities
try:
    from .benchmarking import (
        time_execution,
        benchmark_optimization,
        benchmark_scaling,
        benchmark_hyperparameters
    )
except ImportError:
    # Optional dependencies may not be available
    pass
