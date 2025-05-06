"""
Training algorithms for Decision Jungles.
"""

from .lsearch import LSearch
from .clustersearch import ClusterSearch
from .objective import information_gain, entropy, weighted_entropy_sum, optimize_split

# Try to import Cythonized version
try:
    from .cyth_lsearch import CythLSearch
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
