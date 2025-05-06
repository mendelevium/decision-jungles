#!/bin/bash
# Script to build the distribution packages for PyPI

set -e  # Exit immediately if a command exits with a non-zero status

# Make sure we're in the project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Building distribution packages in $PROJECT_ROOT"

# Check if required tools are installed
for cmd in python pip twine; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is not installed. Please install it with:"
        echo "pip install $cmd"
        exit 1
    fi
done

# Make sure build and twine are installed
pip install --upgrade pip build twine

# Clean up any previous builds
rm -rf build/ dist/ *.egg-info/

# Build the package
echo "Building source distribution and wheel..."
python -m build

# Check the package with twine
echo "Checking the package with twine..."
twine check dist/*

echo "
Distribution packages have been created successfully in the 'dist/' directory.

To test the package locally, run:
  pip install dist/*.whl

To upload to Test PyPI, run:
  twine upload --repository-url https://test.pypi.org/legacy/ dist/*

To upload to PyPI, run:
  twine upload dist/*
"