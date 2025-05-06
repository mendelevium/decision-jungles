"""
Visualization utilities for Decision Jungles.

This module provides functions for visualizing Decision Jungles, including
plotting DAGs, showing node distributions, and visualizing predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set, Any
import networkx as nx
from ..node import Node, SplitNode, LeafNode


def plot_dag(dag: Any, max_depth: Optional[int] = None, figsize: Tuple[int, int] = (12, 8),
            node_size: int = 500, font_size: int = 10, feature_names: Optional[List[str]] = None,
            class_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Plot a DAG from a Decision Jungle using networkx.
    
    Args:
        dag: A fitted DAG.
        max_depth (int, optional): Maximum depth to plot. If None, plot all levels.
        figsize (tuple): Figure size (width, height).
        node_size (int): Size of nodes in the plot.
        font_size (int): Font size for node labels.
        feature_names (List[str], optional): Names of features for better node labeling.
        class_names (List[str], optional): Names of classes for better leaf node labeling.
        
    Returns:
        plt.Figure: The figure containing the plot.
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for node_id, node in dag.nodes.items():
        if max_depth is not None and node.depth > max_depth:
            continue
        
        # Add node with attributes
        if isinstance(node, SplitNode):
            feature_name = f"F{node.feature_idx}"
            if feature_names is not None and 0 <= node.feature_idx < len(feature_names):
                feature_name = feature_names[node.feature_idx]
            label = f"Split\n{feature_name}\nT={node.threshold:.2f}"
            G.add_node(node_id, label=label, node_type='split', depth=node.depth)
        else:  # LeafNode
            # Find the majority class
            if hasattr(node, 'class_distribution') and len(node.class_distribution) > 0:
                majority_class = np.argmax(node.class_distribution)
                ratio = node.class_distribution[majority_class] / node.n_samples if node.n_samples > 0 else 0
                class_label = f"Class {majority_class}"
                if class_names is not None and 0 <= majority_class < len(class_names):
                    class_label = class_names[majority_class]
                label = f"Leaf\n{class_label}\n{ratio:.2f}"
            else:
                label = "Leaf"
            G.add_node(node_id, label=label, node_type='leaf', depth=node.depth)
    
    # Add edges to the graph
    for node_id, node in dag.nodes.items():
        if max_depth is not None and node.depth > max_depth:
            continue
        
        if isinstance(node, SplitNode):
            # Add edges to children
            if node.left_child is not None and (max_depth is None or node.depth + 1 <= max_depth):
                G.add_edge(node_id, node.left_child, edge_type='left')
            if node.right_child is not None and (max_depth is None or node.depth + 1 <= max_depth):
                G.add_edge(node_id, node.right_child, edge_type='right')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get positions for nodes based on hierarchical layout
    pos = nx.multipartite_layout(G, subset_key="depth")
    
    # Draw nodes
    split_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'split']
    leaf_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'leaf']
    
    nx.draw_networkx_nodes(G, pos, nodelist=split_nodes, node_color='skyblue', node_size=node_size, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes, node_color='lightgreen', node_size=node_size, ax=ax)
    
    # Draw edges
    left_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('edge_type') == 'left']
    right_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('edge_type') == 'right']
    
    nx.draw_networkx_edges(G, pos, edgelist=left_edges, edge_color='blue', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=right_edges, edge_color='red', ax=ax)
    
    # Draw labels
    labels = {n: attr['label'] for n, attr in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, ax=ax)
    
    # Add legend
    ax.plot([], [], 'o', color='skyblue', label='Split Node')
    ax.plot([], [], 'o', color='lightgreen', label='Leaf Node')
    ax.plot([], [], '-', color='blue', label='Left Branch')
    ax.plot([], [], '-', color='red', label='Right Branch')
    ax.legend()
    
    plt.title(f"Decision DAG Structure (nodes: {len(G.nodes)}, edges: {len(G.edges)})")
    plt.axis('off')
    plt.tight_layout()
    
    return fig


def plot_class_distribution(node: LeafNode, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot the class distribution at a leaf node.
    
    Args:
        node (LeafNode): The leaf node to visualize.
        figsize (tuple): Figure size (width, height).
        
    Returns:
        plt.Figure: The figure containing the plot.
    """
    if not isinstance(node, LeafNode):
        raise ValueError("Node must be a LeafNode")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get class distribution
    distribution = node.class_distribution
    classes = np.arange(len(distribution))
    
    # Normalize to get probabilities
    if node.n_samples > 0:
        probabilities = distribution / node.n_samples
    else:
        probabilities = np.zeros_like(distribution)
    
    # Plot bar chart
    ax.bar(classes, probabilities, color='skyblue')
    
    # Add labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    ax.set_title(f'Class Distribution at Node {node.node_id} (Depth {node.depth})')
    
    # Set x-ticks to be integers
    ax.set_xticks(classes)
    
    plt.tight_layout()
    
    return fig


def plot_memory_comparison(jungle_memory: int, forest_memory: int, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot a comparison of memory usage between a Decision Jungle and a Random Forest.
    
    Args:
        jungle_memory (int): Memory usage of the Decision Jungle in bytes.
        forest_memory (int): Memory usage of the Random Forest in bytes.
        figsize (tuple): Figure size (width, height).
        
    Returns:
        plt.Figure: The figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to more readable units (MB)
    jungle_memory_mb = jungle_memory / (1024 * 1024)
    forest_memory_mb = forest_memory / (1024 * 1024)
    
    # Plot bar chart
    models = ['Decision Jungle', 'Random Forest']
    memory_usage = [jungle_memory_mb, forest_memory_mb]
    
    ax.bar(models, memory_usage, color=['green', 'blue'])
    
    # Add labels and title
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage Comparison')
    
    # Add values on top of the bars
    for i, v in enumerate(memory_usage):
        ax.text(i, v + 0.1, f"{v:.2f} MB", ha='center')
    
    # Add ratio as text
    ratio = forest_memory / jungle_memory if jungle_memory > 0 else float('inf')
    ax.text(0.5, max(memory_usage) * 0.8, f"Memory Reduction Factor: {ratio:.2f}x", 
            ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig


def plot_accuracy_vs_nodes(results: List[Dict[str, Any]], figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot accuracy vs number of nodes for different models.
    
    Args:
        results (List[Dict]): List of dictionaries containing results for different models.
            Each dictionary should have keys: 'name', 'nodes', 'accuracy'.
        figsize (tuple): Figure size (width, height).
        
    Returns:
        plt.Figure: The figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group results by model name
    models = {}
    for result in results:
        name = result['name']
        if name not in models:
            models[name] = {'nodes': [], 'accuracy': []}
        models[name]['nodes'].append(result['nodes'])
        models[name]['accuracy'].append(result['accuracy'])
    
    # Plot each model
    for name, data in models.items():
        ax.plot(data['nodes'], data['accuracy'], 'o-', label=name)
    
    # Add labels and title
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Number of Nodes')
    
    # Use logarithmic scale for x-axis
    ax.set_xscale('log')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    return fig


def plot_accuracy_vs_evaluations(results: List[Dict[str, Any]], figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot accuracy vs number of feature evaluations for different models.
    
    Args:
        results (List[Dict]): List of dictionaries containing results for different models.
            Each dictionary should have keys: 'name', 'evaluations', 'accuracy'.
        figsize (tuple): Figure size (width, height).
        
    Returns:
        plt.Figure: The figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group results by model name
    models = {}
    for result in results:
        name = result['name']
        if name not in models:
            models[name] = {'evaluations': [], 'accuracy': []}
        models[name]['evaluations'].append(result['evaluations'])
        models[name]['accuracy'].append(result['accuracy'])
    
    # Plot each model
    for name, data in models.items():
        ax.plot(data['evaluations'], data['accuracy'], 'o-', label=name)
    
    # Add labels and title
    ax.set_xlabel('Max. Number of Feature Evaluations per Sample')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Number of Feature Evaluations')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    return fig
