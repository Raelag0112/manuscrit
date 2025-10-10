"""
Graph visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import torch
from torch_geometric.data import Data


def plot_graph_3d(
    data: Data,
    output_path: str = None,
    node_size: int = 50,
    edge_alpha: float = 0.3,
    figsize: tuple = (10, 8),
):
    """
    Plot 3D graph
    
    Args:
        data: PyG Data object
        output_path: Path to save figure (None = show)
        node_size: Size of nodes
        edge_alpha: Edge transparency
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get positions
    pos = data.pos.cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    
    # Plot edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        xs = [pos[src, 0], pos[dst, 0]]
        ys = [pos[src, 1], pos[dst, 1]]
        zs = [pos[src, 2], pos[dst, 2]]
        ax.plot(xs, ys, zs, 'gray', alpha=edge_alpha, linewidth=0.5)
    
    # Plot nodes
    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c='steelblue', s=node_size, alpha=0.8
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Organoid Graph ({data.num_nodes} cells, {data.num_edges} edges)')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_graph_2d(
    data: Data,
    output_path: str = None,
    projection: str = 'xy',
    node_size: int = 50,
):
    """
    Plot 2D projection of graph
    
    Args:
        data: PyG Data object
        output_path: Path to save figure
        projection: 'xy', 'xz', or 'yz'
        node_size: Node size
    """
    plt.figure(figsize=(10, 10))
    
    pos = data.pos.cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    
    # Select projection
    if projection == 'xy':
        x, y = pos[:, 0], pos[:, 1]
    elif projection == 'xz':
        x, y = pos[:, 0], pos[:, 2]
    else:  # yz
        x, y = pos[:, 1], pos[:, 2]
    
    # Plot edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        plt.plot([x[src], x[dst]], [y[src], y[dst]],
                'gray', alpha=0.3, linewidth=0.5)
    
    # Plot nodes
    plt.scatter(x, y, c='steelblue', s=node_size, alpha=0.8)
    
    plt.xlabel(projection[0].upper())
    plt.ylabel(projection[1].upper())
    plt.title(f'Organoid Graph - {projection.upper()} projection')
    plt.axis('equal')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test with dummy data
    num_nodes = 100
    pos = torch.randn(num_nodes, 3) * 50
    edge_index = torch.randint(0, num_nodes, (2, 300))
    
    data = Data(pos=pos, edge_index=edge_index)
    
    plot_graph_3d(data, 'test_graph_3d.png')
    plot_graph_2d(data, 'test_graph_2d.png')
    print("Test plots saved!")

