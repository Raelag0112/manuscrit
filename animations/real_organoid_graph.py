"""
Animation: Real Organoid Graph Visualization
Displays the 3D graph from a real cauliflower organoid with rotation.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

# Configuration
FIG_SIZE = (12, 10)
DPI = 150
SAVE_GIF = True
K_NEIGHBORS = 8  # Number of nearest neighbors for edge construction

# Path to the graph file
GRAPH_PATH = r"D:\data\graph_3_202502_Nice_orga1_5_Chouxfleurs.json"


def load_graph(filepath):
    """Load graph from JSON file."""
    print(f"Loading graph from: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    nodes = data['nodes']
    print(f"Loaded {len(nodes)} nodes")
    
    # Extract coordinates and volumes
    positions = np.array([[n['x'], n['y'], n['z']] for n in nodes])
    volumes = np.array([n['volume'] for n in nodes])
    cell_ids = np.array([n['cell_id'] for n in nodes])
    
    return positions, volumes, cell_ids


def build_knn_edges(positions, k=8, max_distance=None):
    """Build k-nearest neighbor edges."""
    print(f"Building KNN graph with k={k}...")
    tree = KDTree(positions)
    edges = []
    
    for i in range(len(positions)):
        distances, indices = tree.query(positions[i], k=k+1)
        for j, d in zip(indices[1:], distances[1:]):  # Skip self
            if max_distance is None or d < max_distance:
                if i < j:  # Avoid duplicates
                    edges.append((i, j, d))
    
    print(f"Created {len(edges)} edges")
    return edges


def normalize_positions(positions):
    """Center and scale positions."""
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    scale = np.abs(centered).max()
    normalized = centered / scale
    return normalized


def animate_real_organoid():
    """Create animation of the real organoid graph."""
    # Load data
    positions, volumes, cell_ids = load_graph(GRAPH_PATH)
    
    # Normalize
    positions = normalize_positions(positions)
    
    # Normalize volumes for visualization (size of points)
    vol_normalized = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    point_sizes = 10 + vol_normalized * 50  # Size between 10 and 60
    
    # Build edges
    # Calculate a reasonable max distance based on data
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    median_nn_dist = np.median(np.min(distances, axis=1))
    max_edge_dist = median_nn_dist * 3
    
    edges = build_knn_edges(positions, k=K_NEIGHBORS, max_distance=max_edge_dist)
    
    # Setup figure
    fig = plt.figure(figsize=FIG_SIZE)
    fig.patch.set_facecolor('#1a1a2e')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a2e')
    
    # Style 3D axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333333')
    ax.yaxis.pane.set_edgecolor('#333333')
    ax.zaxis.pane.set_edgecolor('#333333')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Set axis limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    
    # Title
    ax.set_title('Cauliflower Organoid\n(Real Data - 3D Graph)', 
                 color='#FF9800', fontsize=14, fontweight='bold', pad=20)
    
    # Color by volume (clustered regions have larger cells)
    colors = plt.cm.YlOrRd(vol_normalized)
    
    # Initialize scatter plot
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=colors, s=point_sizes, alpha=0.8, edgecolors='white', linewidths=0.3)
    
    # Draw edges
    edge_lines = []
    edge_alphas = []
    
    for u, v, d in edges:
        x = [positions[u, 0], positions[v, 0]]
        y = [positions[u, 1], positions[v, 1]]
        z = [positions[u, 2], positions[v, 2]]
        
        # Shorter edges are more visible
        alpha = 0.6 * (1 - d / max_edge_dist) + 0.1
        edge_alphas.append(alpha)
        
        line, = ax.plot(x, y, z, color='#FF9800', alpha=alpha, linewidth=0.5)
        edge_lines.append(line)
    
    # Info text
    info_text = fig.text(0.02, 0.02, 
                        f'Nodes: {len(positions)} | Edges: {len(edges)} | Phenotype: Cauliflower',
                        color='white', fontsize=10, family='monospace')
    
    # Angle text
    angle_text = fig.text(0.98, 0.02, 'Rotation: 0°', 
                         color='#888888', fontsize=10, ha='right')
    
    def init():
        return [scatter] + edge_lines
    
    def animate(frame):
        # Smooth rotation
        azim = frame * 2  # 2 degrees per frame
        elev = 20 + 10 * np.sin(frame * np.pi / 90)  # Gentle up/down motion
        
        ax.view_init(elev=elev, azim=azim)
        
        # Update angle display
        angle_text.set_text(f'Rotation: {azim % 360:.0f}°')
        
        # Subtle pulsing effect on edges every 90 frames
        if frame % 90 < 45:
            pulse = 0.8 + 0.2 * np.sin(frame * 0.1)
        else:
            pulse = 1.0
        
        for line, base_alpha in zip(edge_lines, edge_alphas):
            line.set_alpha(base_alpha * pulse)
        
        return [scatter] + edge_lines
    
    plt.tight_layout()
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=180, interval=50, blit=False)
    
    if SAVE_GIF:
        print("Saving real_organoid_graph.gif...")
        anim.save('real_organoid_graph.gif', writer='pillow', fps=20, dpi=DPI)
        print("Saved!")
    
    plt.close(fig)
    return anim


def create_static_views():
    """Create static multi-view figure of the organoid."""
    # Load data
    positions, volumes, cell_ids = load_graph(GRAPH_PATH)
    positions = normalize_positions(positions)
    
    vol_normalized = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    point_sizes = 10 + vol_normalized * 40
    colors = plt.cm.YlOrRd(vol_normalized)
    
    # Build edges
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    median_nn_dist = np.median(np.min(distances, axis=1))
    max_edge_dist = median_nn_dist * 3
    edges = build_knn_edges(positions, k=K_NEIGHBORS, max_distance=max_edge_dist)
    
    # Create multi-view figure
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    views = [
        ('Front View', 0, 0),
        ('Side View', 0, 90),
        ('Top View', 90, 0),
        ('Isometric', 30, 45)
    ]
    
    for idx, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
        ax.set_facecolor('#1a1a2e')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        
        # Draw edges
        for u, v, d in edges:
            x = [positions[u, 0], positions[v, 0]]
            y = [positions[u, 1], positions[v, 1]]
            z = [positions[u, 2], positions[v, 2]]
            alpha = 0.4 * (1 - d / max_edge_dist) + 0.1
            ax.plot(x, y, z, color='#FF9800', alpha=alpha, linewidth=0.3)
        
        # Draw nodes
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c=colors, s=point_sizes, alpha=0.8, edgecolors='white', linewidths=0.2)
        
        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
    
    plt.suptitle('Cauliflower Organoid - Multi-View Visualization', 
                 color='#FF9800', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('real_organoid_multiview.png', dpi=DPI, 
                facecolor='#1a1a2e', bbox_inches='tight')
    print("Saved real_organoid_multiview.png")
    plt.close(fig)


def create_graph_with_clusters():
    """Create visualization highlighting cluster structure."""
    from scipy.cluster.hierarchy import fcluster, linkage
    
    # Load data
    positions, volumes, cell_ids = load_graph(GRAPH_PATH)
    positions = normalize_positions(positions)
    
    vol_normalized = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    point_sizes = 15 + vol_normalized * 45
    
    # Hierarchical clustering to identify clusters
    print("Performing hierarchical clustering...")
    linkage_matrix = linkage(positions, method='ward')
    n_clusters = 8
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Color by cluster
    cluster_colors = plt.cm.tab10(cluster_labels / n_clusters)
    
    # Build edges
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    median_nn_dist = np.median(np.min(distances, axis=1))
    max_edge_dist = median_nn_dist * 3
    edges = build_knn_edges(positions, k=K_NEIGHBORS, max_distance=max_edge_dist)
    
    # Create figure
    fig = plt.figure(figsize=FIG_SIZE)
    fig.patch.set_facecolor('#1a1a2e')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a2e')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    
    ax.set_title('Cauliflower Organoid - Cluster Analysis\n(Hierarchical Clustering)', 
                 color='#FF9800', fontsize=14, fontweight='bold', pad=20)
    
    # Draw edges colored by cluster
    for u, v, d in edges:
        x = [positions[u, 0], positions[v, 0]]
        y = [positions[u, 1], positions[v, 1]]
        z = [positions[u, 2], positions[v, 2]]
        
        # Color edge by cluster (use color of one endpoint)
        if cluster_labels[u] == cluster_labels[v]:
            color = cluster_colors[u]
            alpha = 0.5
        else:
            color = '#666666'
            alpha = 0.2
        
        ax.plot(x, y, z, color=color, alpha=alpha, linewidth=0.5)
    
    # Draw nodes
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c=cluster_colors, s=point_sizes, alpha=0.9, edgecolors='white', linewidths=0.3)
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig('real_organoid_clusters.png', dpi=DPI, 
                facecolor='#1a1a2e', bbox_inches='tight')
    print("Saved real_organoid_clusters.png")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs('animations', exist_ok=True)
    
    print("=" * 60)
    print("Real Organoid Graph Animation")
    print("=" * 60)
    
    # Generate main animation
    animate_real_organoid()
    
    # Generate static views
    print("\nGenerating static multi-view...")
    create_static_views()
    
    # Generate cluster visualization
    print("\nGenerating cluster visualization...")
    create_graph_with_clusters()
    
    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)
