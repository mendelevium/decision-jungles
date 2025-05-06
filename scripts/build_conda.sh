#!/bin/bash
# Script to build the conda package

set -e  # Exit immediately if a command exits with a non-zero status

# Make sure we're in the project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Building conda package in $PROJECT_ROOT"

# Check if conda-build is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install miniconda or anaconda."
    exit 1
fi

# Install conda-build if needed
conda install -y conda-build conda-verify

# Build the conda package
echo "Building conda package..."
conda build conda-recipe

echo "
Conda package has been built successfully.

To install the package from the local build, run:
  conda install --use-local decision-jungles

To upload to Anaconda Cloud, run:
  anaconda upload /path/to/conda/build/noarch/decision-jungles-0.1.0-py_0.tar.bz2
"