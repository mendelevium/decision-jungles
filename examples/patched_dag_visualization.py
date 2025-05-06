#!/usr/bin/env python
"""
A patched version of the dag_visualization.py example that works with the current codebase.
This version handles feature indices and class indices more safely.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
import networkx as nx
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Import our modules
from decision_jungles import DecisionJungleClassifier
from decision_jungles.utils.visualization import plot_dag

def visualize_decision_path():
    """
    Visualize a decision path for a specific sample.
    """
    print("\nVisualizing Decision Paths")
    print("========================")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train a Decision Jungle
    jungle = DecisionJungleClassifier(
        n_estimators=1,
        max_width=8,
        max_depth=5,
        random_state=42
    )
    
    jungle.fit(X_train, y_train)
    
    # Select a specific sample
    sample_idx = 25  # Modify this index if needed
    if sample_idx < len(X_test):
        sample = X_test[sample_idx]
        true_class = y_test[sample_idx]
    else:
        # Fallback if index is out of range
        sample = X_test[0]
        true_class = y_test[0]
        sample_idx = 0
    
    # Get the DAG
    dag = jungle.dags_[0]
    
    # Since we don't have get_decision_path, let's manually trace a path by simulating prediction
    path = []
    current_node_id = dag.root_node_id
    path.append(current_node_id)
    
    # Trace through the DAG until we reach a leaf node
    while True:
        current_node = dag.nodes[current_node_id]
        
        # If we've reached a leaf node, we're done
        if not hasattr(current_node, 'feature_idx'):
            break
            
        # For split nodes, determine which child to go to
        feature_idx = current_node.feature_idx
        
        # Ensure feature_idx is valid for the sample
        if 0 <= feature_idx < len(sample):
            feature_value = sample[feature_idx]
            
            # Handle categorical features
            if hasattr(current_node, 'is_categorical') and current_node.is_categorical:
                if hasattr(current_node, 'categories_left') and current_node.categories_left:
                    if not np.isnan(feature_value) and feature_value in current_node.categories_left:
                        current_node_id = current_node.left_child
                    else:
                        current_node_id = current_node.right_child
                else:
                    current_node_id = current_node.right_child
            else:
                # Numerical feature
                if np.isnan(feature_value):
                    current_node_id = current_node.right_child
                elif feature_value <= current_node.threshold:
                    current_node_id = current_node.left_child
                else:
                    current_node_id = current_node.right_child
                    
            path.append(current_node_id)
        else:
            # If feature index is invalid, just break
            break
    
    # Create graph visualization
    plt.figure(figsize=(12, 8))
    G = nx.DiGraph()
    
    # Add nodes and edges to the graph
    for node_id in dag.nodes:
        node = dag.nodes[node_id]
        if hasattr(node, 'feature_idx'):  # Split node
            G.add_node(node_id, type='split', feature=node.feature_idx, threshold=node.threshold)
        else:  # Leaf node
            G.add_node(node_id, type='leaf', distribution=node.class_distribution if hasattr(node, 'class_distribution') else None)
    
    # Add edges
    for node_id in dag.nodes:
        node = dag.nodes[node_id]
        if hasattr(node, 'feature_idx'):  # Split node has children
            G.add_edge(node_id, node.left_child, direction='left')
            G.add_edge(node_id, node.right_child, direction='right')
    
    # Create a layout
    pos = nx.spring_layout(G, seed=42)
    
    # Separate path and non-path nodes for coloring
    path_nodes = set(path)
    non_path_nodes = set(G.nodes) - path_nodes
    
    # Get path edges
    path_edges = []
    for i in range(len(path) - 1):
        path_edges.append((path[i], path[i+1]))
    
    non_path_edges = [(u, v) for u, v in G.edges if (u, v) not in path_edges]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n in non_path_nodes if G.nodes[n]['type'] == 'split'],
                          node_color='lightgray',
                          node_size=800,
                          alpha=0.5)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n in non_path_nodes if G.nodes[n]['type'] == 'leaf'],
                          node_color='lightgreen',
                          node_size=800,
                          alpha=0.5)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n in path_nodes if G.nodes[n]['type'] == 'split'],
                          node_color='red',
                          node_size=1000)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n in path_nodes if G.nodes[n]['type'] == 'leaf'],
                          node_color='green',
                          node_size=1000)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edgelist=non_path_edges,
                          edge_color='gray',
                          alpha=0.5,
                          width=1.5,
                          arrowsize=15)
    
    nx.draw_networkx_edges(G, pos, 
                          edgelist=path_edges,
                          edge_color='red',
                          width=2.5,
                          arrowsize=20)
    
    # Add node labels with safe feature access
    labels = {}
    for node in G.nodes:
        if G.nodes[node]['type'] == 'split':
            feature_idx = G.nodes[node]['feature']
            threshold = G.nodes[node]['threshold']
            
            # Safe feature name access
            if 0 <= feature_idx < len(iris.feature_names):
                feature_name = iris.feature_names[feature_idx]
            else:
                feature_name = f"Feature {feature_idx}"
            
            # Safe feature value access
            if 0 <= feature_idx < len(sample):
                feature_value = sample[feature_idx]
                # Add the feature value to the label if this node is in the path
                if node in path_nodes:
                    labels[node] = f"{feature_name}\n≤ {threshold:.2f}\nValue: {feature_value:.2f}"
                else:
                    labels[node] = f"{feature_name}\n≤ {threshold:.2f}"
            else:
                # Fallback when feature index is out of bounds
                labels[node] = f"{feature_name}\n≤ {threshold:.2f}"
        else:
            # For leaf nodes, show the class distribution
            if node in path_nodes and G.nodes[node]['distribution'] is not None:
                class_dist = G.nodes[node]['distribution']
                if len(class_dist) > 0:
                    pred_class = np.argmax(class_dist)
                    # Safe class name access
                    if 0 <= pred_class < len(iris.target_names):
                        labels[node] = f"Leaf {node}\nPredicted: {iris.target_names[pred_class]}"
                    else:
                        labels[node] = f"Leaf {node}\nPredicted: Class {pred_class}"
                else:
                    labels[node] = f"Leaf {node}"
            else:
                labels[node] = f"Leaf {node}"
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # Add title and legend
    class_name = iris.target_names[true_class] if 0 <= true_class < len(iris.target_names) else f"Class {true_class}"
    plt.title(f"Decision Path for Sample {sample_idx}\nTrue Class: {class_name}")
    
    # Create legend
    legend_elements = [
        Patch(facecolor='red', label='Path Split Node'),
        Patch(facecolor='green', label='Path Leaf Node'),
        Patch(facecolor='lightgray', alpha=0.5, label='Other Split Node'),
        Patch(facecolor='lightgreen', alpha=0.5, label='Other Leaf Node'),
        Line2D([0], [0], color='red', lw=2.5, label='Path Edge'),
        Line2D([0], [0], color='gray', lw=1.5, alpha=0.5, label='Other Edge')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    plt.savefig('visualizations/patched_decision_path.png')
    plt.close()
    
    print("Decision path visualization saved as 'visualizations/patched_decision_path.png'")
    
    # Visualize feature values
    plt.figure(figsize=(10, 6))
    
    # Ensure we only plot valid feature indices
    valid_indices = min(len(sample), len(iris.feature_names))
    valid_sample = sample[:valid_indices]
    valid_feature_names = iris.feature_names[:valid_indices]
    
    # Create a horizontal bar chart of feature values
    plt.barh(range(len(valid_sample)), valid_sample, color='skyblue')
    plt.yticks(range(len(valid_sample)), valid_feature_names)
    plt.xlabel('Feature Value')
    plt.title(f'Feature Values for Sample {sample_idx} (Class: {class_name})')
    
    # Add a vertical line for mean feature values
    mean_values = np.mean(X, axis=0)[:valid_indices]
    for i, mean_val in enumerate(mean_values):
        plt.plot([mean_val, mean_val], [i-0.4, i+0.4], 'r--', alpha=0.7)
    
    # Add legend
    plt.plot([], [], 'r--', label='Mean Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/patched_sample_features.png')
    plt.close()
    
    print("Sample feature visualization saved as 'visualizations/patched_sample_features.png'")

def visualize_basic_dag():
    """
    Visualize a basic Decision Jungle DAG trained on the Iris dataset.
    """
    print("Basic DAG Visualization")
    print("======================")
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train a small Decision Jungle
    jungle = DecisionJungleClassifier(
        n_estimators=1,  # Just one DAG for visualization
        max_width=8,     # Small width for better visualization
        max_depth=4,     # Limited depth for better visualization
        random_state=42
    )
    
    jungle.fit(X_train, y_train)
    
    # Get the single DAG
    dag = jungle.dags_[0]
    
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Visualize the DAG structure using modified plot_dag that accepts feature_names
    # Since our updated plot_dag now supports feature_names, we can use it directly
    fig = plot_dag(dag, max_depth=4, figsize=(14, 10), node_size=1000, font_size=8)
    
    # Add custom feature names to the plot
    plt.savefig('visualizations/patched_basic_dag.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Basic DAG visualization saved as 'visualizations/patched_basic_dag.png'")

if __name__ == "__main__":
    visualize_basic_dag()
    visualize_decision_path()