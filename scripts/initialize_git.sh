#!/bin/bash
# Script to initialize git repository for Decision Jungles

set -e  # Exit immediately if a command exits with a non-zero status

# Make sure we're in the project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Initializing git repository in $PROJECT_ROOT"

# Check if git is already initialized
if [ -d ".git" ]; then
    echo "Git repository already exists."
    exit 0
fi

# Initialize git repository
git init

# Create .gitignore
cat > .gitignore << EOL
# Python bytecode
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/
*.egg

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
coverage.xml
*.cover
.pytest_cache/

# Virtual environments
venv/
env/
ENV/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# Compiled extensions
*.so
*.pyd
*.dll

# Jupyter Notebook
.ipynb_checkpoints

# Project specific
undo/
EOL

# Add all files
git add -A

# Make initial commit
git commit -m "Initial commit

This is the initial commit for the Decision Jungles project, a scikit-learn 
compatible implementation of Decision Jungles as described in the paper 
'Decision Jungles: Compact and Rich Models for Classification' by Jamie Shotton 
et al. (NIPS 2013)."

# Instructions for remote repository
echo "
Git repository has been initialized.

To connect this repository to a remote repository on GitHub, run:

git remote add origin https://github.com/yourusername/decision-jungles.git
git branch -M main
git push -u origin main
"

echo "Git initialization completed successfully!"