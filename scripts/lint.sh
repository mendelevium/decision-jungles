#!/bin/bash
# Script to run linting and code style checks

set -e  # Exit immediately if a command exits with a non-zero status

# Make sure we're in the project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Running code quality checks in $PROJECT_ROOT"

# Check if required tools are installed
for cmd in flake8 black isort mypy; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is not installed. Please install it with:"
        echo "pip install $cmd"
        exit 1
    fi
done

# Check Python syntax errors and undefined names
echo "1. Running flake8 to check for syntax errors and undefined names..."
flake8 decision_jungles tests

# Format code with black
echo "2. Checking code formatting with black..."
black --check decision_jungles tests

# Check import sorting
echo "3. Checking import sorting with isort..."
isort --check-only --profile black decision_jungles tests

# Run static type checking
echo "4. Running static type checking with mypy..."
mypy decision_jungles

echo "All code quality checks passed!"