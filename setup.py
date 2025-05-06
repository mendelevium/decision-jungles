from setuptools import setup, find_packages

setup(
    name="decision-jungles",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.3.0",
        "scikit-learn>=0.21.0",
        "matplotlib>=3.1.0",
        "networkx>=2.3",
        "joblib>=0.13.0",
    ],
    extras_require={
        "performance": ["cython>=0.29.0"],
        "profiling": [
            "psutil>=5.9.0",
            "memory-profiler>=0.60.0",
            "pympler>=1.0.1",
            "pandas>=1.3.0",
            "tabulate>=0.8.0",
        ],
        "dev": [
            "pytest>=5.0.0",
            "hypothesis>=6.0.0",
        ],
    },
    author="Decision Jungle Team",
    author_email="info@example.com",
    description="A scikit-learn compatible implementation of Decision Jungles",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/decision-jungles",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Machine Learning",
    ],
    python_requires=">=3.8",
)
