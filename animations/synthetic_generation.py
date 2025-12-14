"""
Animation: Synthetic Data Generation
Shows the process of generating synthetic organoids from a sphere using point processes.
Demonstrates Poisson (cystic) vs Matérn cluster (cauliflower) generation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Configuration
np.random.seed(42)
FIG_SIZE = (14, 7)
DPI = 150
SAVE_GIF = True
N_FRAMES = 240


def generate_sphere_wireframe(radius=1.0, n_lines=20):
    """Generate wireframe sphere coordinates."""
    # Longitude lines
    phi = np.linspace(0, np.pi, 50)
    theta_lines = np.linspace(0, 2*np.pi, n_lines, endpoint=False)
    
    lon_lines = []
    for theta in theta_lines:
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        lon_lines.append((x, y, z))
    
    # Latitude lines
    theta = np.linspace(0, 2*np.pi, 100)
    phi_lines = np.linspace(0.2, np.pi-0.2, 8)
    
    lat_lines = []
    for p in phi_lines:
        x = radius * np.sin(p) * np.cos(theta)
        y = radius * np.sin(p) * np.sin(theta)
        z = radius * np.cos(p) * np.ones_like(theta)
        lat_lines.append((x, y, z))
    
    return lon_lines, lat_lines


def generate_poisson_points(n_points, radius=1.0, shell_ratio=0.8):
    """Generate Poisson (uniform) distributed points in a spherical shell."""
    points = []
    while len(points) < n_points:
        # Uniform in spherical shell
        r = radius * np.cbrt(np.random.uniform(shell_ratio**3, 1.0))
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * np.random.uniform() - 1)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        points.append([x, y, z])
    
    return np.array(points)


def generate_matern_points(n_parents, n_children_mean, radius=1.0, cluster_sigma=0.12):
    """Generate Matérn cluster process points."""
    parent_points = []
    child_points = []
    parent_indices = []  # Track which parent each child belongs to
    
    # Generate parent points on sphere surface
    for i in range(n_parents):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * np.random.uniform() - 1)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        parent_points.append([x, y, z])
    
    # Generate children around each parent
    for i, parent in enumerate(parent_points):
        n_children = max(1, np.random.poisson(n_children_mean))
        for _ in range(n_children):
            # Add Gaussian displacement
            offset = np.random.normal(0, cluster_sigma, 3)
            child = np.array(parent) + offset
            
            # Project back to spherical shell
            norm = np.linalg.norm(child)
            if norm > 0:
                child = child / norm * radius * np.random.uniform(0.85, 1.0)
            
            child_points.append(child)
            parent_indices.append(i)
    
    return np.array(parent_points), np.array(child_points), np.array(parent_indices)


def animate_synthetic_generation():
    """Create animation showing synthetic data generation."""
    fig = plt.figure(figsize=FIG_SIZE)
    fig.patch.set_facecolor('#1a1a2e')
    
    # Two subplots: Poisson (cystic) and Matérn (cauliflower)
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
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-1.3, 1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    # Titles
    title1 = ax1.set_title('Cystic Organoid\n(Poisson Process)', 
                           color='#4CAF50', fontsize=12, fontweight='bold', pad=10)
    title2 = ax2.set_title('Cauliflower Organoid\n(Matérn Cluster Process)', 
                           color='#FF9800', fontsize=12, fontweight='bold', pad=10)
    
    # Generate sphere wireframe
    lon_lines, lat_lines = generate_sphere_wireframe()
    
    # Store wireframe line objects
    wireframe1 = []
    wireframe2 = []
    
    for x, y, z in lon_lines + lat_lines:
        line1, = ax1.plot(x, y, z, color='#4CAF50', alpha=0, linewidth=0.5)
        line2, = ax2.plot(x, y, z, color='#FF9800', alpha=0, linewidth=0.5)
        wireframe1.append(line1)
        wireframe2.append(line2)
    
    # Generate all points upfront
    n_poisson = 120
    poisson_points = generate_poisson_points(n_poisson)
    
    n_parents = 8
    parent_points, child_points, parent_indices = generate_matern_points(n_parents, 15)
    
    # Colors for parents
    parent_colors = plt.cm.tab10(np.linspace(0, 1, n_parents))
    child_colors = parent_colors[parent_indices]
    
    # Initialize scatter plots (empty)
    scatter1 = ax1.scatter([], [], [], c='#4CAF50', s=40, alpha=0.8)
    scatter2_parents = ax2.scatter([], [], [], c='red', s=150, alpha=0.9, marker='^', edgecolors='white', linewidths=1)
    scatter2_children = ax2.scatter([], [], [], c='#FF9800', s=40, alpha=0.8)
    
    # Phase text
    phase_text = fig.text(0.5, 0.02, '', ha='center', fontsize=11, color='white')
    
    # Stats text
    stats1 = fig.text(0.25, 0.92, '', ha='center', fontsize=10, color='#4CAF50')
    stats2 = fig.text(0.75, 0.92, '', ha='center', fontsize=10, color='#FF9800')
    
    def animate(frame):
        # Rotation
        azim = frame * 1.5
        ax1.view_init(elev=20, azim=azim)
        ax2.view_init(elev=20, azim=azim)
        
        # Phase 1: Draw sphere wireframe (frames 0-40)
        if frame < 40:
            progress = frame / 40
            phase_text.set_text('Step 1: Define spherical domain')
            
            # Fade in wireframe
            for line in wireframe1 + wireframe2:
                line.set_alpha(progress * 0.4)
            
            stats1.set_text('')
            stats2.set_text('')
        
        # Phase 2: Show Poisson points appearing (frames 40-100)
        elif frame < 100:
            progress = (frame - 40) / 60
            phase_text.set_text('Step 2: Generate cell positions')
            
            # Poisson: points appear uniformly
            n_visible = int(n_poisson * progress)
            if n_visible > 0:
                scatter1._offsets3d = (poisson_points[:n_visible, 0],
                                       poisson_points[:n_visible, 1],
                                       poisson_points[:n_visible, 2])
            
            # Matérn: first show parent points
            n_parents_visible = int(n_parents * min(1, progress * 2))
            if n_parents_visible > 0:
                scatter2_parents._offsets3d = (parent_points[:n_parents_visible, 0],
                                               parent_points[:n_parents_visible, 1],
                                               parent_points[:n_parents_visible, 2])
                scatter2_parents.set_facecolors(parent_colors[:n_parents_visible])
            
            # Then children appear around parents
            if progress > 0.3:
                child_progress = (progress - 0.3) / 0.7
                n_children_visible = int(len(child_points) * child_progress)
                if n_children_visible > 0:
                    scatter2_children._offsets3d = (child_points[:n_children_visible, 0],
                                                    child_points[:n_children_visible, 1],
                                                    child_points[:n_children_visible, 2])
                    scatter2_children.set_facecolors(child_colors[:n_children_visible])
            
            stats1.set_text(f'Cells: {n_visible}')
            n_total_matern = n_parents_visible + (int(len(child_points) * max(0, (progress - 0.3) / 0.7)) if progress > 0.3 else 0)
            stats2.set_text(f'Parents: {n_parents_visible} | Children: {n_total_matern - n_parents_visible}')
        
        # Phase 3: Fade out wireframe, show final result (frames 100-160)
        elif frame < 160:
            progress = (frame - 100) / 60
            phase_text.set_text('Step 3: Complete synthetic organoid')
            
            # Fade out wireframe
            for line in wireframe1 + wireframe2:
                line.set_alpha(0.4 * (1 - progress))
            
            # All points visible
            scatter1._offsets3d = (poisson_points[:, 0],
                                   poisson_points[:, 1],
                                   poisson_points[:, 2])
            
            scatter2_parents._offsets3d = (parent_points[:, 0],
                                           parent_points[:, 1],
                                           parent_points[:, 2])
            scatter2_parents.set_facecolors(parent_colors)
            
            scatter2_children._offsets3d = (child_points[:, 0],
                                            child_points[:, 1],
                                            child_points[:, 2])
            scatter2_children.set_facecolors(child_colors)
            
            # Fade out parent markers
            scatter2_parents.set_alpha(1 - progress * 0.7)
            
            stats1.set_text(f'Cells: {n_poisson} | Distribution: Uniform')
            stats2.set_text(f'Cells: {len(child_points)} | Clusters: {n_parents}')
        
        # Phase 4: Rotation showcase (frames 160+)
        else:
            phase_text.set_text('Synthetic organoids ready for training!')
            
            # Pulsing effect
            pulse = 0.8 + 0.2 * np.sin((frame - 160) * 0.15)
            scatter1.set_alpha(pulse)
            scatter2_children.set_alpha(pulse)
            
            stats1.set_text(f'Cells: {n_poisson} | Distribution: Uniform')
            stats2.set_text(f'Cells: {len(child_points)} | Clusters: {n_parents}')
        
        return [scatter1, scatter2_parents, scatter2_children, phase_text, stats1, stats2] + wireframe1 + wireframe2
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08)
    
    anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=50, blit=False)
    
    if SAVE_GIF:
        print("Saving synthetic_generation.gif...")
        anim.save('synthetic_generation.gif', writer='pillow', fps=20, dpi=DPI)
        print("Saved!")
    
    plt.close(fig)
    return anim


def create_generation_process_figure():
    """Create a static figure showing the generation process step by step."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    # 2 rows x 4 columns
    # Row 1: Poisson process steps
    # Row 2: Matérn process steps
    
    steps_poisson = [
        ('1. Sphere Domain', 'sphere'),
        ('2. Random Sampling', 'partial'),
        ('3. Uniform Points', 'full'),
        ('4. Cystic Organoid', 'final')
    ]
    
    steps_matern = [
        ('1. Sphere Domain', 'sphere'),
        ('2. Parent Points', 'parents'),
        ('3. Children Clusters', 'children'),
        ('4. Cauliflower Organoid', 'final')
    ]
    
    # Generate data
    poisson_points = generate_poisson_points(100)
    parent_points, child_points, parent_indices = generate_matern_points(6, 15)
    parent_colors = plt.cm.tab10(np.linspace(0, 1, len(parent_points)))
    child_colors = parent_colors[parent_indices]
    
    lon_lines, lat_lines = generate_sphere_wireframe(n_lines=12)
    
    for row, (steps, color, points_data) in enumerate([
        (steps_poisson, '#4CAF50', poisson_points),
        (steps_matern, '#FF9800', (parent_points, child_points, parent_indices, parent_colors, child_colors))
    ]):
        for col, (title, step_type) in enumerate(steps):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection='3d')
            ax.set_facecolor('#1a1a2e')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_zlim(-1.3, 1.3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=20, azim=45)
            
            ax.set_title(title, color=color, fontsize=10, fontweight='bold', pad=5)
            
            # Draw based on step type
            if step_type == 'sphere':
                for x, y, z in lon_lines + lat_lines:
                    ax.plot(x, y, z, color=color, alpha=0.5, linewidth=0.5)
            
            elif step_type == 'partial':
                for x, y, z in lon_lines + lat_lines:
                    ax.plot(x, y, z, color=color, alpha=0.2, linewidth=0.5)
                n = len(points_data) // 3
                ax.scatter(points_data[:n, 0], points_data[:n, 1], points_data[:n, 2],
                          c=color, s=30, alpha=0.8)
            
            elif step_type == 'full':
                ax.scatter(points_data[:, 0], points_data[:, 1], points_data[:, 2],
                          c=color, s=30, alpha=0.8)
            
            elif step_type == 'parents':
                for x, y, z in lon_lines + lat_lines:
                    ax.plot(x, y, z, color=color, alpha=0.2, linewidth=0.5)
                pp, cp, pi, pc, cc = points_data
                ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2],
                          c=pc, s=150, alpha=0.9, marker='^', edgecolors='white', linewidths=1)
            
            elif step_type == 'children':
                pp, cp, pi, pc, cc = points_data
                ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2],
                          c=pc, s=100, alpha=0.7, marker='^', edgecolors='white', linewidths=0.5)
                ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2],
                          c=cc, s=30, alpha=0.8)
            
            elif step_type == 'final':
                if row == 0:
                    ax.scatter(points_data[:, 0], points_data[:, 1], points_data[:, 2],
                              c=color, s=40, alpha=0.9, edgecolors='white', linewidths=0.3)
                else:
                    pp, cp, pi, pc, cc = points_data
                    ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2],
                              c=cc, s=40, alpha=0.9, edgecolors='white', linewidths=0.3)
    
    # Row labels
    fig.text(0.02, 0.75, 'POISSON\n(Cystic)', ha='left', va='center', 
             fontsize=12, fontweight='bold', color='#4CAF50', rotation=90)
    fig.text(0.02, 0.25, 'MATÉRN\n(Cauliflower)', ha='left', va='center', 
             fontsize=12, fontweight='bold', color='#FF9800', rotation=90)
    
    plt.suptitle('Synthetic Organoid Generation Process', 
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.savefig('synthetic_generation_steps.png', dpi=DPI, 
                facecolor='#1a1a2e', bbox_inches='tight')
    print("Saved synthetic_generation_steps.png")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("Synthetic Data Generation Animation")
    print("=" * 60)
    
    # Generate main animation
    animate_synthetic_generation()
    
    # Generate static step-by-step figure
    print("\nGenerating step-by-step figure...")
    create_generation_process_figure()
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)
