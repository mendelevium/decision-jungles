"""
Setup script for compiling Cython extensions for Decision Jungles.

Run this script to compile the Cython extensions:
python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "decision_jungles.training.cyth_lsearch",
        sources=["decision-jungles/training/cyth_lsearch.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-O3"]
    )
]

setup(
    name="decision_jungles_cython",
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()]
)