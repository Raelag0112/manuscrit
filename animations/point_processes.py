"""
Animation: Point Processes Comparison
Shows the difference between Poisson (cystic) and Matérn cluster (cauliflower) processes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Configuration
np.random.seed(42)
FIG_SIZE = (14, 6)
DPI = 150
SAVE_GIF = True


def generate_poisson_sphere(n_points, radius=1.0):
    """Generate uniformly distributed points on a sphere (Poisson process)."""
    points = []
    while len(points) < n_points:
        # Uniform in spherical shell
        r = radius * np.cbrt(np.random.uniform(0.7, 1.0))  # Shell
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * np.random.uniform() - 1)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        points.append([x, y, z])
    
    return np.array(points)


def generate_matern_sphere(n_parents, n_children_per_parent, radius=1.0, cluster_radius=0.15):
    """Generate Matérn cluster process on a sphere (cauliflower pattern)."""
    points = []
    parent_points = []
    
    # Generate parent points on sphere surface
    for _ in range(n_parents):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * np.random.uniform() - 1)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        parent_points.append([x, y, z])
    
    # Generate children around each parent
    for parent in parent_points:
        n_children = np.random.poisson(n_children_per_parent)
        for _ in range(max(1, n_children)):
            # Add Gaussian displacement
            offset = np.random.normal(0, cluster_radius, 3)
            child = np.array(parent) + offset
            
            # Project back to spherical shell
            norm = np.linalg.norm(child)
            if norm > 0:
                child = child / norm * radius * np.random.uniform(0.85, 1.0)
            
            points.append(child)
    
    return np.array(points), np.array(parent_points)


def animate_point_processes():
    """Create animation comparing Poisson and Matérn processes."""
    fig = plt.figure(figsize=FIG_SIZE)
    fig.patch.set_facecolor('#1a1a2e')
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1a1a2e')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333333')
        ax.yaxis.pane.set_edgecolor('#333333')
        ax.zaxis.pane.set_edgecolor('#333333')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    ax1.set_title('Cystic (Poisson Process)\nUniform Distribution', 
                  color='#4CAF50', fontsize=12, fontweight='bold', pad=10)
    ax2.set_title('Cauliflower (Matérn Cluster)\nClustered Distribution', 
                  color='#FF9800', fontsize=12, fontweight='bold', pad=10)
    
    # Generate points
    poisson_points = generate_poisson_sphere(150)
    matern_points, parent_points = generate_matern_sphere(8, 18, cluster_radius=0.12)
    
    # Initialize scatter plots
    scatter1 = ax1.scatter([], [], [], c='#4CAF50', s=30, alpha=0.8)
    scatter2 = ax2.scatter([], [], [], c='#FF9800', s=30, alpha=0.8)
    scatter_parents = ax2.scatter([], [], [], c='#F44336', s=100, alpha=0.9, marker='^')
    
    def init():
        return scatter1, scatter2, scatter_parents
    
    def animate(frame):
        # Rotation angle
        angle = frame * 2
        ax1.view_init(elev=20, azim=angle)
        ax2.view_init(elev=20, azim=angle)
        
        # Progressive point appearance
        if frame < 60:
            n_visible = int(len(poisson_points) * frame / 60)
            n_visible_m = int(len(matern_points) * frame / 60)
            n_parents = int(len(parent_points) * frame / 30)
        else:
            n_visible = len(poisson_points)
            n_visible_m = len(matern_points)
            n_parents = len(parent_points)
        
        # Update Poisson
        if n_visible > 0:
            scatter1._offsets3d = (poisson_points[:n_visible, 0],
                                   poisson_points[:n_visible, 1],
                                   poisson_points[:n_visible, 2])
        
        # Update Matérn
        if n_visible_m > 0:
            scatter2._offsets3d = (matern_points[:n_visible_m, 0],
                                   matern_points[:n_visible_m, 1],
                                   matern_points[:n_visible_m, 2])
        
        # Update parent points
        if n_parents > 0 and frame < 90:
            scatter_parents._offsets3d = (parent_points[:n_parents, 0],
                                          parent_points[:n_parents, 1],
                                          parent_points[:n_parents, 2])
        elif frame >= 90:
            # Fade out parent points
            alpha = max(0, 1 - (frame - 90) / 30)
            scatter_parents.set_alpha(alpha)
            scatter_parents._offsets3d = (parent_points[:, 0],
                                          parent_points[:, 1],
                                          parent_points[:, 2])
        
        return scatter1, scatter2, scatter_parents
    
    plt.tight_layout()
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=180, interval=50, blit=False)
    
    if SAVE_GIF:
        print("Saving point_processes.gif...")
        anim.save('animations/point_processes.gif', writer='pillow', fps=20, dpi=DPI)
        print("Saved!")
    
    plt.show()
    return anim


if __name__ == "__main__":
    import os
    os.makedirs('animations', exist_ok=True)
    animate_point_processes()
