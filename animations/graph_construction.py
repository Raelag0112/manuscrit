"""
Animation: Graph Construction from Point Cloud
Shows the transformation from cell positions to geometric graph.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import KDTree

# Configuration
np.random.seed(123)
FIG_SIZE = (12, 5)
DPI = 150
SAVE_GIF = True


def generate_organoid_cells(n_cells=50):
    """Generate cell positions mimicking an organoid cross-section."""
    cells = []
    
    # Outer ring (more dense)
    for i in range(int(n_cells * 0.6)):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.7, 1.0)
        x = r * np.cos(angle) + np.random.normal(0, 0.05)
        y = r * np.sin(angle) + np.random.normal(0, 0.05)
        cells.append([x, y])
    
    # Inner region (less dense)
    for i in range(int(n_cells * 0.4)):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.2, 0.6)
        x = r * np.cos(angle) + np.random.normal(0, 0.05)
        y = r * np.sin(angle) + np.random.normal(0, 0.05)
        cells.append([x, y])
    
    return np.array(cells)


def build_knn_graph(points, k=5):
    """Build k-nearest neighbor graph."""
    tree = KDTree(points)
    edges = []
    
    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=k+1)  # +1 because point itself is included
        for j in indices[1:]:  # Skip self
            if i < j:  # Avoid duplicates
                edges.append((i, j, distances[list(indices).index(j)]))
    
    return edges


def animate_graph_construction():
    """Create animation showing graph construction process."""
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    axes[0].set_title('1. Cell Detection', color='white', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_title('2. Feature Extraction', color='white', fontsize=12, fontweight='bold', pad=10)
    axes[2].set_title('3. Graph Construction (KNN)', color='white', fontsize=12, fontweight='bold', pad=10)
    
    # Generate cells
    cells = generate_organoid_cells(40)
    volumes = np.random.uniform(0.5, 1.5, len(cells))  # Simulated volumes
    
    # Build graph
    edges = build_knn_graph(cells, k=4)
    
    # Initialize plots
    # Panel 1: Raw cells appearing
    scatter1 = axes[0].scatter([], [], c='#4CAF50', s=50, alpha=0.8)
    
    # Panel 2: Cells with features (size = volume)
    scatter2 = axes[1].scatter([], [], c=[], s=[], alpha=0.8, cmap='viridis')
    
    # Panel 3: Graph
    scatter3 = axes[2].scatter([], [], c='#2196F3', s=50, alpha=0.8)
    edge_lines = []
    
    # Add colorbar for panel 2
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0.5, vmax=1.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Cell Volume', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    def init():
        return [scatter1, scatter2, scatter3]
    
    def animate(frame):
        # Clear previous edge lines
        for line in edge_lines:
            line.remove()
        edge_lines.clear()
        
        # Phase 1: Cells appearing (frames 0-40)
        if frame < 40:
            n_visible = int(len(cells) * (frame + 1) / 40)
            
            # Panel 1: Raw detection
            scatter1.set_offsets(cells[:n_visible])
            scatter1.set_sizes([50] * n_visible)
            
            # Panel 2: Empty during detection
            scatter2.set_offsets(np.empty((0, 2)))
            
            # Panel 3: Empty
            scatter3.set_offsets(np.empty((0, 2)))
        
        # Phase 2: Feature extraction (frames 40-80)
        elif frame < 80:
            progress = (frame - 40) / 40
            n_processed = int(len(cells) * progress)
            
            # Panel 1: All cells visible
            scatter1.set_offsets(cells)
            colors1 = ['#4CAF50' if i >= n_processed else '#666666' for i in range(len(cells))]
            scatter1.set_facecolors(colors1)
            
            # Panel 2: Processed cells with features
            if n_processed > 0:
                scatter2.set_offsets(cells[:n_processed])
                scatter2.set_sizes(volumes[:n_processed] * 80)
                scatter2.set_array(volumes[:n_processed])
            
            # Panel 3: Empty
            scatter3.set_offsets(np.empty((0, 2)))
        
        # Phase 3: Graph construction (frames 80-150)
        elif frame < 150:
            progress = (frame - 80) / 70
            n_edges = int(len(edges) * progress)
            
            # Panel 1: All cells (faded)
            scatter1.set_offsets(cells)
            scatter1.set_facecolors(['#666666'] * len(cells))
            
            # Panel 2: All features
            scatter2.set_offsets(cells)
            scatter2.set_sizes(volumes * 80)
            scatter2.set_array(volumes)
            
            # Panel 3: Graph with edges
            scatter3.set_offsets(cells)
            
            # Draw edges
            for i, (u, v, d) in enumerate(edges[:n_edges]):
                x = [cells[u, 0], cells[v, 0]]
                y = [cells[u, 1], cells[v, 1]]
                
                # Color based on distance
                alpha = 0.8 - 0.3 * (d / max(e[2] for e in edges))
                line, = axes[2].plot(x, y, color='#FF9800', alpha=alpha, linewidth=1.5, zorder=1)
                edge_lines.append(line)
        
        # Phase 4: Final state with pulsing (frames 150+)
        else:
            pulse = 0.8 + 0.2 * np.sin((frame - 150) * 0.2)
            
            # Panel 1: Faded
            scatter1.set_offsets(cells)
            scatter1.set_facecolors(['#333333'] * len(cells))
            
            # Panel 2: All features
            scatter2.set_offsets(cells)
            scatter2.set_sizes(volumes * 80)
            scatter2.set_array(volumes)
            
            # Panel 3: Complete graph
            scatter3.set_offsets(cells)
            scatter3.set_sizes([50 * pulse] * len(cells))
            
            for u, v, d in edges:
                x = [cells[u, 0], cells[v, 0]]
                y = [cells[u, 1], cells[v, 1]]
                alpha = (0.8 - 0.3 * (d / max(e[2] for e in edges))) * pulse
                line, = axes[2].plot(x, y, color='#FF9800', alpha=alpha, linewidth=1.5, zorder=1)
                edge_lines.append(line)
        
        return [scatter1, scatter2, scatter3] + edge_lines
    
    plt.tight_layout()
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=180, interval=50, blit=False)
    
    if SAVE_GIF:
        print("Saving graph_construction.gif...")
        anim.save('animations/graph_construction.gif', writer='pillow', fps=20, dpi=DPI)
        print("Saved!")
    
    plt.show()
    return anim


if __name__ == "__main__":
    import os
    os.makedirs('animations', exist_ok=True)
    animate_graph_construction()
