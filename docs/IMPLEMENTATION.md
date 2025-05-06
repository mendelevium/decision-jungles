# Decision Jungles Implementation Details

This document provides detailed information about the implementation of Decision Jungles as described in the paper ["Decision Jungles: Compact and Rich Models for Classification"](https://www.microsoft.com/en-us/research/publication/decision-jungles-compact-and-rich-models-for-classification/) by Jamie Shotton et al. (NIPS 2013).

## Overview

Decision Jungles are ensembles of rooted decision directed acyclic graphs (DAGs) that offer two key advantages over traditional decision trees/forests:

1. **Reduced memory footprint** through node merging
2. **Improved generalization** through regularization effects of the DAG structure

Unlike conventional decision trees that only allow one path to every node, a DAG in a decision jungle allows multiple paths from the root to each leaf. This results in a more compact model with potentially better generalization.

## Key Components

### 1. Node Classes

Two primary node types form the building blocks of our Decision Jungles:

#### SplitNode

- Internal nodes that contain a binary split function
- Each split node has:
  - A feature index and threshold for splitting
  - References to left and right child nodes
  - A set of parent nodes (multiple parents possible in a DAG)

#### LeafNode

- Terminal nodes that provide predictions
- Each leaf node has:
  - A class distribution for classification
  - A set of parent nodes (multiple parents possible in a DAG)

### 2. DAG Structure

The Directed Acyclic Graph (DAG) is a fundamental structure that:

- Contains a collection of nodes (both split and leaf nodes)
- Has a single root node but allows multiple paths to any node
- Tracks parent-child relationships for all nodes
- Manages the flow of samples during both training and prediction

### 3. Optimization Algorithms

Two optimization methods are implemented for training the DAGs:

#### LSearch

The primary optimization method, which alternates between:

1. **Split Optimization**: Finding the best split parameters for each parent node
2. **Branch Optimization**: Redirecting branches to optimize the objective function

LSearch operates in a coordinate-descent manner, optimizing one aspect at a time until convergence.

#### ClusterSearch

An alternative optimization method that:

1. Builds temporary nodes via conventional tree-based training
2. Clusters the temporary nodes based on class distribution similarity
3. Merges similar nodes to form a more compact DAG

### 4. Merging Schedules

Different merging schedules control the width of each level in the DAG:

- **Constant**: Fixed maximum width at each level
- **Exponential**: Width grows exponentially with depth up to max_width
- **Kinect**: Special schedule used for Kinect body part classification (as described in the paper)

### 5. Objective Function

The decision jungles use an information gain objective function based on:

- Shannon entropy for class distributions
- Weighted sum of entropies across multiple nodes
- Information gain from splitting

## Training Process

The training process for Decision Jungles follows these steps:

1. **Initialization**: Create a root node with all training samples
2. **Level-by-level Growth**:
   - Determine the width of the next level based on the merging schedule
   - Create temporary leaf nodes for the next level
   - Use the selected optimization method (LSearch or ClusterSearch) to:
     - Find optimal split parameters for each parent node
     - Determine optimal branch assignments to child nodes
   - Update the DAG structure and route samples accordingly
3. **Stopping Criteria**:
   - Maximum depth reached
   - No more nodes can be split
   - All samples have been perfectly classified

## Prediction Process

Prediction in a Decision Jungle involves:

1. Start all samples at the root node
2. Route samples through the DAG according to split functions
3. When a sample reaches a leaf node, use the class distribution for prediction
4. Combine predictions from all DAGs in the ensemble by averaging

## Memory Efficiency

The memory efficiency of Decision Jungles comes from:

1. **Node Merging**: Multiple paths can lead to the same node, reducing duplication
2. **Width Control**: The merging schedule limits the width at each level
3. **Efficient Storage**: Only essential information is stored for each node

## Performance Considerations

Several optimizations improve the performance:

1. **Vectorized Operations**: Using NumPy for efficient computations
2. **Efficient Sample Routing**: Tracking sample indices rather than copying data
3. **Parallel Training**: Option to train DAGs in parallel using joblib

## Implementation Challenges

Some of the challenges addressed in this implementation:

1. **DAG Structure Management**: Tracking both parent and child relationships
2. **Optimization Convergence**: Ensuring the LSearch algorithm converges efficiently
3. **Sample Routing**: Efficiently routing samples through the DAG during training
4. **Merging Strategy**: Implementing effective node merging while maintaining accuracy

## Comparison with the Paper

Our implementation follows the key concepts from the original paper:

- The level-by-level growth of DAGs
- The LSearch and ClusterSearch optimization methods
- The merging schedules, including the special Kinect schedule
- The objective function based on weighted entropy sum

Some minor implementation differences:

- We use a scikit-learn compatible interface for easier integration
- We've added more flexibility in hyperparameters
- Our implementation includes parallel training options

## Future Improvements

Potential areas for future enhancement:

1. **Regression Support**: Extend the implementation to regression tasks
2. **Merging Between DAGs**: Implement merging between DAGs in a jungle
3. **Multiply Rooted Trees**: Explore the use of multiply rooted trees as mentioned in the paper
4. **Feature Importance**: Add support for feature importance calculation
5. **Performance Optimizations**: Further optimize computation-intensive parts
