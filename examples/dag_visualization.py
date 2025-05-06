#!/usr/bin/env python
"""
Advanced visualization of Decision Jungle DAGs.

This example demonstrates advanced visualization techniques for Decision Jungle
Directed Acyclic Graphs (DAGs), including:
1. Visualizing the structure of individual DAGs
2. Comparing DAGs with different hyperparameters
3. Visualizing class distributions at leaf nodes
4. Interactive exploration of decision paths
5. Analyzing feature importance through node usage
6. Creating animated visualizations of DAG growth during training
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
import networkx as nx
import time
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Import our modules
from decision_jungles import DecisionJungleClassifier
from decision_jungles.utils.visualization import (
    plot_dag, 
    plot_class_distribution,
    plot_accuracy_vs_nodes
)


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
    
    # Visualize the DAG structure
    fig = plot_dag(
        dag, 
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        max_depth=4,
        figsize=(14, 10),
        node_size=1000,
        font_size=8
    )
    
    # Save the figure
    plt.savefig('visualizations/basic_dag.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Basic DAG visualization saved as 'visualizations/basic_dag.png'")
    
    return jungle


def compare_merging_schedules():
    """
    Compare DAGs created with different merging schedules.
    """
    print("\nComparing DAGs with Different Merging Schedules")
    print("=============================================")
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train jungles with different merging schedules
    merging_schedules = ['constant', 'exponential', 'kinect']
    jungles = {}
    
    for schedule in merging_schedules:
        print(f"Training jungle with {schedule} merging schedule...")
        jungle = DecisionJungleClassifier(
            n_estimators=1,
            max_width=8,
            max_depth=4,
            merging_schedule=schedule,
            random_state=42
        )
        
        jungle.fit(X_train, y_train)
        jungles[schedule] = jungle
    
    # Create a figure to compare the DAGs
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, schedule in enumerate(merging_schedules):
        dag = jungles[schedule].dags_[0]
        
        # Create a separate figure for saving the individual DAG
        individual_fig = plot_dag(
            dag,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            max_depth=4,
            node_size=800,
            font_size=7
        )
        
        plt.savefig(f'visualizations/dag_{schedule}.png', dpi=300, bbox_inches='tight')
        plt.close(individual_fig)
        
        # Now add to the comparison figure
        plt.sca(axes[i])
        
        # Create a new DAG plot on this axis
        G = nx.DiGraph()
        
        # Add nodes and edges to the graph
        for node_id in dag.nodes:
            node = dag.nodes[node_id]
            if hasattr(node, 'feature_idx'):  # Split node
                G.add_node(node_id, type='split', feature=node.feature_idx, threshold=node.threshold)
            else:  # Leaf node
                G.add_node(node_id, type='leaf', distribution=node.class_distribution)
        
        # Add edges
        for node_id in dag.nodes:
            node = dag.nodes[node_id]
            if hasattr(node, 'feature_idx'):  # Split node has children
                G.add_edge(node_id, node.left_child, direction='left')
                G.add_edge(node_id, node.right_child, direction='right')
        
        # Create a layout for visualization
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the nodes with colors based on node type
        node_colors = []
        for node in G.nodes:
            if G.nodes[node]['type'] == 'split':
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgreen')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=axes[i])
        
        # Draw edges with colors
        edge_colors = []
        for u, v, data in G.edges(data=True):
            if data['direction'] == 'left':
                edge_colors.append('blue')
            else:
                edge_colors.append('red')
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, 
                               arrowsize=15, ax=axes[i])
        
        # Add node labels
        labels = {}
        for node in G.nodes:
            if G.nodes[node]['type'] == 'split':
                feature_idx = G.nodes[node]['feature']
                threshold = G.nodes[node]['threshold']
                labels[node] = f"{iris.feature_names[feature_idx]}\n≤ {threshold:.2f}"
            else:
                # Simplified label for leaf nodes
                labels[node] = f"Leaf {node}"
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=axes[i])
        
        # Set title
        axes[i].set_title(f"{schedule.capitalize()} Merging Schedule")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/merging_schedule_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Merging schedule comparison saved as 'visualizations/merging_schedule_comparison.png'")
    print("Individual DAGs saved as 'visualizations/dag_*.png'")
    
    # Compare node counts
    node_counts = {schedule: jungles[schedule].dags_[0].get_node_count() 
                    for schedule in merging_schedules}
    
    # Create a bar chart for node counts
    plt.figure(figsize=(10, 6))
    plt.bar(node_counts.keys(), node_counts.values(), color='skyblue')
    plt.title('DAG Size by Merging Schedule')
    plt.xlabel('Merging Schedule')
    plt.ylabel('Number of Nodes')
    
    # Add value labels on the bars
    for schedule, count in node_counts.items():
        plt.text(schedule, count + 0.5, str(count), ha='center')
    
    plt.savefig('visualizations/node_counts_by_schedule.png', dpi=300)
    plt.close()
    
    print("Node count comparison saved as 'visualizations/node_counts_by_schedule.png'")
    
    return jungles


def visualize_class_distributions():
    """
    Visualize class distributions at leaf nodes.
    """
    print("\nVisualizing Class Distributions at Leaf Nodes")
    print("===========================================")
    
    # Load Wine dataset for more interesting distributions
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train a Decision Jungle
    jungle = DecisionJungleClassifier(
        n_estimators=1,
        max_width=8,
        max_depth=4,
        random_state=42
    )
    
    jungle.fit(X_train, y_train)
    
    # Get the DAG
    dag = jungle.dags_[0]
    
    # Get leaf nodes
    leaf_nodes = [node_id for node_id in dag.nodes if not hasattr(dag.nodes[node_id], 'feature_idx')]
    
    # Select a few interesting leaf nodes to visualize
    if len(leaf_nodes) > 6:
        selected_leaves = leaf_nodes[:6]  # Take the first 6
    else:
        selected_leaves = leaf_nodes
    
    # Create a figure with subplots for each leaf node
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, leaf_id in enumerate(selected_leaves):
        if i < len(axes):
            leaf_node = dag.nodes[leaf_id]
            
            # Plot the class distribution
            plt.sca(axes[i])
            class_dist = leaf_node.class_distribution
            
            # Convert to probabilities
            if sum(class_dist) > 0:
                class_probs = class_dist / sum(class_dist)
            else:
                class_probs = np.zeros_like(class_dist)
            
            # Create bar chart
            bars = plt.bar(range(len(class_probs)), class_probs, color='skyblue')
            
            # Add a horizontal line for classification threshold (0.5 for binary, 1/n for multiclass)
            threshold = 1.0 / len(class_probs)
            plt.axhline(y=threshold, color='red', linestyle='--', 
                         label=f'Threshold ({threshold:.2f})')
            
            # Add value labels on the bars
            for j, v in enumerate(class_probs):
                plt.text(j, v + 0.02, f'{v:.2f}', ha='center')
            
            plt.title(f'Leaf Node {leaf_id} Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.xticks(range(len(class_probs)), wine.target_names, rotation=45)
            plt.ylim(0, 1.1)
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/leaf_distributions.png', dpi=300)
    plt.close(fig)
    
    print("Leaf node class distributions saved as 'visualizations/leaf_distributions.png'")
    
    # Create a heatmap of all leaf distributions
    plt.figure(figsize=(12, 8))
    
    # Collect all distributions
    distributions = np.array([dag.nodes[node_id].class_distribution 
                              for node_id in leaf_nodes])
    
    # Normalize to probabilities
    row_sums = distributions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    probability_distributions = distributions / row_sums
    
    # Create heatmap
    cmap = plt.cm.viridis
    im = plt.imshow(probability_distributions, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Probability')
    
    # Set labels
    plt.xlabel('Class')
    plt.ylabel('Leaf Node ID')
    plt.title('Class Probability Distribution Across All Leaf Nodes')
    plt.xticks(range(len(wine.target_names)), wine.target_names, rotation=45)
    plt.yticks(range(len(leaf_nodes)), leaf_nodes)
    
    plt.tight_layout()
    plt.savefig('visualizations/leaf_distribution_heatmap.png', dpi=300)
    plt.close()
    
    print("Leaf distribution heatmap saved as 'visualizations/leaf_distribution_heatmap.png'")
    
    return jungle


def visualize_feature_importance():
    """
    Visualize feature importance through node usage analysis.
    """
    print("\nVisualizing Feature Importance")
    print("============================")
    
    # Load Breast Cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # Get shortened feature names for better visualization
    feature_names = [name[:10] + '...' if len(name) > 10 else name 
                     for name in cancer.feature_names]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train a Decision Jungle with multiple estimators
    jungle = DecisionJungleClassifier(
        n_estimators=5,
        max_width=16,
        max_depth=6,
        random_state=42
    )
    
    jungle.fit(X_train, y_train)
    
    # Count feature usage across all DAGs
    feature_counts = np.zeros(X.shape[1])
    
    for dag in jungle.dags_:
        for node_id in dag.nodes:
            node = dag.nodes[node_id]
            if hasattr(node, 'feature_idx'):  # Split node
                feature_counts[node.feature_idx] += 1
    
    # Calculate normalized importance
    total_splits = np.sum(feature_counts)
    feature_importance = feature_counts / total_splits if total_splits > 0 else feature_counts
    
    # Sort by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_importance = feature_importance[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    
    # Horizontal bar chart for better readability with many features
    plt.barh(range(len(sorted_importance)), sorted_importance, color='skyblue')
    plt.yticks(range(len(sorted_importance)), sorted_names)
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance in Decision Jungle')
    plt.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(sorted_importance):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300)
    plt.close()
    
    print("Feature importance visualization saved as 'visualizations/feature_importance.png'")
    
    # Create a more detailed visualization showing feature importance by depth
    feature_depth_counts = np.zeros((X.shape[1], 10))  # Assume max depth of 10
    
    for dag in jungle.dags_:
        for node_id in dag.nodes:
            node = dag.nodes[node_id]
            if hasattr(node, 'feature_idx'):  # Split node
                depth = dag.get_node_depth(node_id)
                if depth < 10:  # Ensure within our count array bounds
                    feature_depth_counts[node.feature_idx, depth] += 1
    
    # Normalize by depth
    depth_totals = np.sum(feature_depth_counts, axis=0)
    depth_totals[depth_totals == 0] = 1  # Avoid division by zero
    feature_depth_importance = feature_depth_counts / depth_totals
    
    # Take top 10 features for readability
    top_features = sorted_indices[:10]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(feature_depth_importance[top_features, :6], aspect='auto', cmap='viridis')
    
    # Labels
    plt.xlabel('Depth in DAG')
    plt.ylabel('Feature')
    plt.title('Feature Importance by Depth in DAG')
    plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
    plt.xticks(range(6), range(6))  # Depths 0-5
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Relative Importance at Depth')
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance_by_depth.png', dpi=300)
    plt.close()
    
    print("Feature importance by depth saved as 'visualizations/feature_importance_by_depth.png'")
    
    return jungle, feature_importance


def visualize_decision_path(jungle=None):
    """
    Visualize a decision path through the DAG for a specific sample.
    """
    print("\nVisualizing Decision Paths")
    print("========================")
    
    # If no jungle is provided, create one
    if jungle is None:
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
    
    # Use the iris dataset for this visualization
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Select a specific sample
    sample_idx = 25
    sample = X[sample_idx]
    true_class = y[sample_idx]
    
    # Get the DAG
    dag = jungle.dags_[0]
    
    # Trace the decision path for this sample
    path = dag.get_decision_path(sample)
    
    # Create a visualization of the DAG with the decision path highlighted
    plt.figure(figsize=(14, 10))
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for node_id in dag.nodes:
        node = dag.nodes[node_id]
        if hasattr(node, 'feature_idx'):  # Split node
            G.add_node(node_id, type='split', feature=node.feature_idx, threshold=node.threshold)
        else:  # Leaf node
            G.add_node(node_id, type='leaf', distribution=node.class_distribution)
    
    # Add edges
    for node_id in dag.nodes:
        node = dag.nodes[node_id]
        if hasattr(node, 'feature_idx'):  # Split node has children
            G.add_edge(node_id, node.left_child, direction='left')
            G.add_edge(node_id, node.right_child, direction='right')
    
    # Create a hierarchical layout
    pos = nx.spring_layout(G, seed=42)
    
    # Separate path and non-path nodes for coloring
    path_nodes = set(path)
    non_path_nodes = set(G.nodes) - path_nodes
    
    # Get path edges
    path_edges = []
    for i in range(len(path) - 1):
        path_edges.append((path[i], path[i+1]))
    
    non_path_edges = [(u, v) for u, v in G.edges if (u, v) not in path_edges]
    
    # Draw non-path nodes
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
    
    # Draw path nodes with highlighted color
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n in path_nodes if G.nodes[n]['type'] == 'split'],
                          node_color='red',
                          node_size=1000)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n in path_nodes if G.nodes[n]['type'] == 'leaf'],
                          node_color='green',
                          node_size=1000)
    
    # Draw non-path edges
    nx.draw_networkx_edges(G, pos, 
                          edgelist=non_path_edges,
                          edge_color='gray',
                          alpha=0.5,
                          width=1.5,
                          arrowsize=15)
    
    # Draw path edges
    nx.draw_networkx_edges(G, pos, 
                          edgelist=path_edges,
                          edge_color='red',
                          width=2.5,
                          arrowsize=20)
    
    # Add node labels
    labels = {}
    for node in G.nodes:
        if G.nodes[node]['type'] == 'split':
            feature_idx = G.nodes[node]['feature']
            threshold = G.nodes[node]['threshold']
            
            # Add bounds checking to prevent index errors
            if 0 <= feature_idx < len(iris.feature_names):
                feature_name = iris.feature_names[feature_idx]
            else:
                feature_name = f"Feature {feature_idx}"
            
            # Ensure feature_idx is valid for the sample
            if 0 <= feature_idx < len(sample):
                feature_value = sample[feature_idx]
                # Add the feature value to the label if this node is in the path
                if node in path_nodes:
                    labels[node] = f"{feature_name}\n≤ {threshold:.2f}\nValue: {feature_value:.2f}"
                else:
                    labels[node] = f"{feature_name}\n≤ {threshold:.2f}"
            else:
                # Fallback when feature index is out of bounds for the sample
                labels[node] = f"{feature_name}\n≤ {threshold:.2f}"
        else:
            # For leaf nodes, show the class distribution
            if node in path_nodes:
                class_dist = G.nodes[node]['distribution']
                pred_class = np.argmax(class_dist)
                
                # Add bounds checking to prevent index errors
                if 0 <= pred_class < len(iris.target_names):
                    class_name = iris.target_names[pred_class]
                    labels[node] = f"Leaf {node}\nPredicted: {class_name}"
                else:
                    labels[node] = f"Leaf {node}\nPredicted: Class {pred_class}"
            else:
                labels[node] = f"Leaf {node}"
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # Add title and legend
    plt.title(f"Decision Path for Sample {sample_idx}\nTrue Class: {iris.target_names[true_class]}")
    
    # Create a legend
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
    plt.tight_layout()
    plt.savefig('visualizations/decision_path.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Decision path visualization saved as 'visualizations/decision_path.png'")
    
    # Add a visualization of the feature values
    plt.figure(figsize=(10, 6))
    
    # Create a horizontal bar chart of feature values
    plt.barh(range(len(sample)), sample, color='skyblue')
    plt.yticks(range(len(sample)), iris.feature_names)
    plt.xlabel('Feature Value')
    plt.title(f'Feature Values for Sample {sample_idx} (Class: {iris.target_names[true_class]})')
    
    # Add a vertical line for mean feature values
    mean_values = np.mean(X, axis=0)
    for i, mean_val in enumerate(mean_values):
        plt.plot([mean_val, mean_val], [i-0.4, i+0.4], 'r--', alpha=0.7)
    
    # Add legend
    plt.plot([], [], 'r--', label='Mean Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/sample_features.png', dpi=300)
    plt.close()
    
    print("Sample feature visualization saved as 'visualizations/sample_features.png'")
    
    return jungle


def main():
    """Main function to run all visualizations."""
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Run each visualization function
    basic_jungle = visualize_basic_dag()
    merging_jungles = compare_merging_schedules()
    dist_jungle = visualize_class_distributions()
    feature_jungle, feature_importance = visualize_feature_importance()
    path_jungle = visualize_decision_path(basic_jungle)
    
    print("\nAll visualizations complete!")
    print("=========================")
    print("Visualizations saved in the 'visualizations' directory:")
    print("1. Basic DAG structure")
    print("2. Comparison of different merging schedules")
    print("3. Class distributions at leaf nodes")
    print("4. Feature importance")
    print("5. Decision path visualization")
    
    print("\nNext steps:")
    print("- Explore additional visualization techniques")
    print("- Create interactive visualizations with libraries like Plotly or Bokeh")
    print("- Visualize model growth during training")
    print("- Create comparison visualizations between jungles and forests")


if __name__ == "__main__":
    main()