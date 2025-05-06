"""
Decision Jungles.

A scikit-learn compatible implementation of Decision Jungles as described in
"Decision Jungles: Compact and Rich Models for Classification" by Jamie Shotton et al.

This implementation includes both classification and regression capabilities.
"""

from .jungle import DecisionJungleClassifier
from .regression import DecisionJungleRegressor
from ._version import __version__, get_version, check_version_compatibility

# Import utility functions that should be part of the public API
from .utils import (
    setup_logging,
    get_logger,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
)
