"""
Animation: Graph Construction from 3D Cell Point Cloud
Shows the complete transformation pipeline: 3D image → cells → features → graph
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

# Configuration
np.random.seed(42)
FIG_SIZE = (16, 9)
DPI = 150
SAVE_GIF = True
N_FRAMES = 300


def generate_organoid_3d(n_cells=80, phenotype='cystic'):
    """Generate 3D cell positions mimicking an organoid."""
    cells = []
    volumes = []
    
    if phenotype == 'cystic':
        # Spherical shell distribution (cystic)
        for _ in range(n_cells):
            # Shell distribution
            r = np.random.uniform(0.75, 1.0)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.arccos(2 * np.random.uniform() - 1)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            # Add some noise
            x += np.random.normal(0, 0.03)
            y += np.random.normal(0, 0.03)
            z += np.random.normal(0, 0.03)
            
            cells.append([x, y, z])
            volumes.append(np.random.uniform(0.5, 1.5))
    else:
        # Clustered distribution (cauliflower)
        n_clusters = 6
        cluster_centers = []
        for _ in range(n_clusters):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.arccos(2 * np.random.uniform() - 1)
            center = np.array([np.sin(phi) * np.cos(theta),
                              np.sin(phi) * np.sin(theta),
                              np.cos(phi)])
            cluster_centers.append(center)
        
        for _ in range(n_cells):
            # Pick a random cluster
            center = cluster_centers[np.random.randint(n_clusters)]
            # Generate point near cluster
            offset = np.random.normal(0, 0.15, 3)
            point = center + offset
            # Normalize to shell
            point = point / np.linalg.norm(point) * np.random.uniform(0.8, 1.0)
            cells.append(point)
            volumes.append(np.random.uniform(0.5, 2.0))
    
    return np.array(cells), np.array(volumes)


def build_knn_graph(positions, k=6):
    """Build k-nearest neighbor graph."""
    tree = KDTree(positions)
    edges = []
    
    for i in range(len(positions)):
        distances, indices = tree.query(positions[i], k=k+1)
        for j, d in zip(indices[1:], distances[1:]):
            if i < j:
                edges.append((i, j, d))
    
    return edges


def animate_graph_construction():
    """Create comprehensive 3D graph construction animation."""
    fig = plt.figure(figsize=FIG_SIZE)
    fig.patch.set_facecolor('#0d1117')
    
    # Create grid: main 3D view + side panels
    gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 1], height_ratios=[1, 1],
                         hspace=0.15, wspace=0.2)
    
    # Main 3D view (spans left side)
    ax_main = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Side panels (2D projections and info)
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[0, 2])
    ax_yz = fig.add_subplot(gs[0, 3])
    ax_info = fig.add_subplot(gs[1, 1:])
    
    # Style main 3D axis
    ax_main.set_facecolor('#0d1117')
    ax_main.xaxis.pane.fill = False
    ax_main.yaxis.pane.fill = False
    ax_main.zaxis.pane.fill = False
    ax_main.xaxis.pane.set_edgecolor('#333333')
    ax_main.yaxis.pane.set_edgecolor('#333333')
    ax_main.zaxis.pane.set_edgecolor('#333333')
    ax_main.set_xlim(-1.3, 1.3)
    ax_main.set_ylim(-1.3, 1.3)
    ax_main.set_zlim(-1.3, 1.3)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_zticks([])
    
    # Style 2D projection axes
    for ax, title in [(ax_xy, 'XY Projection'), (ax_xz, 'XZ Projection'), (ax_yz, 'YZ Projection')]:
        ax.set_facecolor('#0d1117')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color='#888888', fontsize=9, pad=5)
        for spine in ax.spines.values():
            spine.set_color('#333333')
    
    # Style info panel
    ax_info.set_facecolor('#0d1117')
    ax_info.axis('off')
    
    # Generate data
    cells, volumes = generate_organoid_3d(70, 'cystic')
    edges = build_knn_graph(cells, k=5)
    
    # Normalize volumes for visualization
    vol_norm = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    point_sizes = 30 + vol_norm * 70
    
    # Color map
    cmap = plt.cm.viridis
    colors = cmap(vol_norm)
    
    # Initialize scatter plots
    scatter_main = ax_main.scatter([], [], [], s=[], c=[], alpha=0.9)
    scatter_xy = ax_xy.scatter([], [], s=[], c=[], alpha=0.7)
    scatter_xz = ax_xz.scatter([], [], s=[], c=[], alpha=0.7)
    scatter_yz = ax_yz.scatter([], [], s=[], c=[], alpha=0.7)
    
    # Edge storage
    edge_lines_3d = []
    edge_lines_xy = []
    edge_lines_xz = []
    edge_lines_yz = []
    
    # Title and phase
    main_title = fig.suptitle('', fontsize=16, fontweight='bold', color='white', y=0.98)
    
    # Info text elements
    info_texts = {
        'phase': ax_info.text(0.5, 0.85, '', ha='center', va='top', fontsize=14, 
                             color='white', fontweight='bold', transform=ax_info.transAxes),
        'step1': ax_info.text(0.1, 0.6, '', ha='left', va='top', fontsize=11, 
                             color='#4CAF50', transform=ax_info.transAxes, family='monospace'),
        'step2': ax_info.text(0.1, 0.4, '', ha='left', va='top', fontsize=11, 
                             color='#2196F3', transform=ax_info.transAxes, family='monospace'),
        'step3': ax_info.text(0.1, 0.2, '', ha='left', va='top', fontsize=11, 
                             color='#FF9800', transform=ax_info.transAxes, family='monospace'),
        'stats': ax_info.text(0.7, 0.5, '', ha='left', va='center', fontsize=10, 
                             color='#888888', transform=ax_info.transAxes, family='monospace'),
    }
    
    # Progress bar background
    progress_bg = plt.Rectangle((0.1, 0.05), 0.8, 0.08, fill=True, 
                                 facecolor='#333333', transform=ax_info.transAxes)
    ax_info.add_patch(progress_bg)
    progress_fill = plt.Rectangle((0.1, 0.05), 0, 0.08, fill=True, 
                                   facecolor='#4CAF50', transform=ax_info.transAxes)
    ax_info.add_patch(progress_fill)
    
    def clear_edges():
        for line_list in [edge_lines_3d, edge_lines_xy, edge_lines_xz, edge_lines_yz]:
            for line in line_list:
                line.remove()
            line_list.clear()
    
    def draw_edges(n_edges, alpha_mult=1.0):
        clear_edges()
        max_dist = max(e[2] for e in edges)
        
        for u, v, d in edges[:n_edges]:
            alpha = (0.7 - 0.4 * (d / max_dist)) * alpha_mult
            lw = 1.5 - 0.8 * (d / max_dist)
            
            # 3D
            line, = ax_main.plot([cells[u, 0], cells[v, 0]],
                                [cells[u, 1], cells[v, 1]],
                                [cells[u, 2], cells[v, 2]],
                                color='#FF9800', alpha=alpha, linewidth=lw)
            edge_lines_3d.append(line)
            
            # XY
            line, = ax_xy.plot([cells[u, 0], cells[v, 0]],
                              [cells[u, 1], cells[v, 1]],
                              color='#FF9800', alpha=alpha*0.5, linewidth=lw*0.7)
            edge_lines_xy.append(line)
            
            # XZ
            line, = ax_xz.plot([cells[u, 0], cells[v, 0]],
                              [cells[u, 2], cells[v, 2]],
                              color='#FF9800', alpha=alpha*0.5, linewidth=lw*0.7)
            edge_lines_xz.append(line)
            
            # YZ
            line, = ax_yz.plot([cells[u, 1], cells[v, 1]],
                              [cells[u, 2], cells[v, 2]],
                              color='#FF9800', alpha=alpha*0.5, linewidth=lw*0.7)
            edge_lines_yz.append(line)
    
    def animate(frame):
        # Rotation
        azim = frame * 0.8
        ax_main.view_init(elev=20, azim=azim)
        
        # ============ PHASE 1: Raw 3D Image (frames 0-50) ============
        if frame < 50:
            progress = frame / 50
            main_title.set_text('Step 1: 3D Confocal Image Acquisition')
            
            info_texts['phase'].set_text('📷 Acquiring 3D confocal image...')
            info_texts['step1'].set_text(f'► Image size: 2048 × 2048 × 200 slices')
            info_texts['step2'].set_text(f'  Data volume: ~2 GB')
            info_texts['step3'].set_text('')
            info_texts['stats'].set_text(f'Slice: {int(progress * 200)}/200')
            
            progress_fill.set_width(0.8 * progress * 0.2)
            progress_fill.set_facecolor('#4CAF50')
            
            # Show cells appearing slice by slice
            n_visible = int(len(cells) * progress)
            if n_visible > 0:
                scatter_main._offsets3d = (cells[:n_visible, 0],
                                          cells[:n_visible, 1],
                                          cells[:n_visible, 2])
                scatter_main.set_sizes(point_sizes[:n_visible] * 0.5)
                scatter_main.set_facecolors(['#666666'] * n_visible)
            
            clear_edges()
            scatter_xy.set_offsets(np.empty((0, 2)))
            scatter_xz.set_offsets(np.empty((0, 2)))
            scatter_yz.set_offsets(np.empty((0, 2)))
        
        # ============ PHASE 2: Cell Segmentation (frames 50-100) ============
        elif frame < 100:
            progress = (frame - 50) / 50
            main_title.set_text('Step 2: Cell Segmentation (Faster Cellpose)')
            
            info_texts['phase'].set_text('🔬 Segmenting individual cells...')
            info_texts['step1'].set_text(f'✓ Image acquired')
            info_texts['step2'].set_text(f'► Running Faster Cellpose (5× faster)')
            info_texts['step3'].set_text(f'  F1 Score: 0.95')
            info_texts['stats'].set_text(f'Cells detected: {int(len(cells) * progress)}/{len(cells)}')
            
            progress_fill.set_width(0.8 * (0.2 + progress * 0.25))
            progress_fill.set_facecolor('#4CAF50')
            
            # Cells being segmented (color transition)
            n_segmented = int(len(cells) * progress)
            scatter_main._offsets3d = (cells[:, 0], cells[:, 1], cells[:, 2])
            
            cell_colors = []
            for i in range(len(cells)):
                if i < n_segmented:
                    cell_colors.append(colors[i])
                else:
                    cell_colors.append([0.4, 0.4, 0.4, 0.5])
            scatter_main.set_facecolors(cell_colors)
            scatter_main.set_sizes(point_sizes * 0.7)
            
            # Update 2D projections
            if n_segmented > 0:
                scatter_xy.set_offsets(cells[:n_segmented, :2])
                scatter_xy.set_sizes(point_sizes[:n_segmented] * 0.4)
                scatter_xy.set_facecolors(colors[:n_segmented])
                
                scatter_xz.set_offsets(cells[:n_segmented, [0, 2]])
                scatter_xz.set_sizes(point_sizes[:n_segmented] * 0.4)
                scatter_xz.set_facecolors(colors[:n_segmented])
                
                scatter_yz.set_offsets(cells[:n_segmented, 1:])
                scatter_yz.set_sizes(point_sizes[:n_segmented] * 0.4)
                scatter_yz.set_facecolors(colors[:n_segmented])
        
        # ============ PHASE 3: Feature Extraction (frames 100-150) ============
        elif frame < 150:
            progress = (frame - 100) / 50
            main_title.set_text('Step 3: Feature Extraction')
            
            info_texts['phase'].set_text('📊 Extracting cell features...')
            info_texts['step1'].set_text(f'✓ Image acquired')
            info_texts['step2'].set_text(f'✓ {len(cells)} cells segmented')
            info_texts['step3'].set_text(f'► Features: (x, y, z, volume)')
            
            feature_list = ['Position (x, y, z)', 'Volume', 'Centroid computed', 'Normalization']
            current_feature = int(progress * len(feature_list))
            info_texts['stats'].set_text(f'Feature: {feature_list[min(current_feature, len(feature_list)-1)]}')
            
            progress_fill.set_width(0.8 * (0.45 + progress * 0.2))
            progress_fill.set_facecolor('#2196F3')
            
            # All cells visible with features
            scatter_main._offsets3d = (cells[:, 0], cells[:, 1], cells[:, 2])
            scatter_main.set_facecolors(colors)
            
            # Pulse effect on sizes to show feature extraction
            pulse = 1 + 0.2 * np.sin(progress * np.pi * 4)
            scatter_main.set_sizes(point_sizes * pulse)
            
            # Full 2D projections
            scatter_xy.set_offsets(cells[:, :2])
            scatter_xy.set_sizes(point_sizes * 0.4)
            scatter_xy.set_facecolors(colors)
            
            scatter_xz.set_offsets(cells[:, [0, 2]])
            scatter_xz.set_sizes(point_sizes * 0.4)
            scatter_xz.set_facecolors(colors)
            
            scatter_yz.set_offsets(cells[:, 1:])
            scatter_yz.set_sizes(point_sizes * 0.4)
            scatter_yz.set_facecolors(colors)
        
        # ============ PHASE 4: Graph Construction (frames 150-230) ============
        elif frame < 230:
            progress = (frame - 150) / 80
            main_title.set_text('Step 4: K-Nearest Neighbors Graph Construction')
            
            info_texts['phase'].set_text('🔗 Building geometric graph...')
            info_texts['step1'].set_text(f'✓ Image acquired')
            info_texts['step2'].set_text(f'✓ {len(cells)} cells segmented')
            info_texts['step3'].set_text(f'✓ Features: (x, y, z, volume)')
            
            n_edges_visible = int(len(edges) * progress)
            info_texts['stats'].set_text(f'K=5 neighbors\nEdges: {n_edges_visible}/{len(edges)}')
            
            progress_fill.set_width(0.8 * (0.65 + progress * 0.3))
            progress_fill.set_facecolor('#FF9800')
            
            # Cells visible
            scatter_main._offsets3d = (cells[:, 0], cells[:, 1], cells[:, 2])
            scatter_main.set_facecolors(colors)
            scatter_main.set_sizes(point_sizes)
            
            # Draw edges progressively
            draw_edges(n_edges_visible)
            
            # 2D projections
            scatter_xy.set_offsets(cells[:, :2])
            scatter_xz.set_offsets(cells[:, [0, 2]])
            scatter_yz.set_offsets(cells[:, 1:])
        
        # ============ PHASE 5: Complete Graph (frames 230+) ============
        else:
            loop_frame = (frame - 230) % 70
            main_title.set_text('Complete: Geometric Graph Ready for GNN')
            
            info_texts['phase'].set_text('✅ Graph construction complete!')
            info_texts['step1'].set_text(f'✓ Nodes: {len(cells)} cells')
            info_texts['step2'].set_text(f'✓ Edges: {len(edges)} connections')
            info_texts['step3'].set_text(f'✓ Features: 4D (x, y, z, volume)')
            info_texts['stats'].set_text(f'Compression:\n2 GB → 10 MB\n(1000× reduction)')
            
            progress_fill.set_width(0.8 * 0.95)
            progress_fill.set_facecolor('#4CAF50')
            
            # Gentle pulsing
            pulse = 0.9 + 0.1 * np.sin(loop_frame * 0.15)
            
            scatter_main._offsets3d = (cells[:, 0], cells[:, 1], cells[:, 2])
            scatter_main.set_facecolors(colors)
            scatter_main.set_sizes(point_sizes * pulse)
            
            draw_edges(len(edges), alpha_mult=pulse)
            
            # 2D projections
            scatter_xy.set_offsets(cells[:, :2])
            scatter_xz.set_offsets(cells[:, [0, 2]])
            scatter_yz.set_offsets(cells[:, 1:])
        
        return [scatter_main, scatter_xy, scatter_xz, scatter_yz, main_title]
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=50, blit=False)
    
    if SAVE_GIF:
        print("Saving graph_construction.gif...")
        anim.save('graph_construction.gif', writer='pillow', fps=20, dpi=DPI)
        print("Saved!")
    
    plt.close(fig)
    return anim


def create_pipeline_figure():
    """Create a static figure showing the full pipeline."""
    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor('#0d1117')
    
    steps = [
        ('3D Confocal\nImage', '2 GB', '#666666'),
        ('Cell\nSegmentation', 'Faster Cellpose', '#4CAF50'),
        ('Feature\nExtraction', '(x,y,z,vol)', '#2196F3'),
        ('Graph\nConstruction', 'K-NN', '#FF9800'),
        ('GNN\nClassification', 'GAT/EGNN', '#9C27B0'),
    ]
    
    # Generate sample data for visualization
    cells, volumes = generate_organoid_3d(50)
    vol_norm = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    colors = plt.cm.viridis(vol_norm)
    edges = build_knn_graph(cells, k=4)
    
    for i, (title, subtitle, color) in enumerate(steps):
        ax = fig.add_subplot(1, 5, i + 1, projection='3d')
        ax.set_facecolor('#0d1117')
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
        
        ax.set_title(f'{title}\n({subtitle})', color=color, fontsize=10, fontweight='bold', pad=10)
        
        if i == 0:  # Raw image (gray points)
            ax.scatter(cells[:, 0], cells[:, 1], cells[:, 2], c='#555555', s=20, alpha=0.5)
        elif i == 1:  # Segmentation (colored cells emerging)
            ax.scatter(cells[:, 0], cells[:, 1], cells[:, 2], c=colors, s=30, alpha=0.8)
        elif i == 2:  # Features (sized by volume)
            sizes = 20 + vol_norm * 50
            ax.scatter(cells[:, 0], cells[:, 1], cells[:, 2], c=colors, s=sizes, alpha=0.9)
        elif i == 3:  # Graph
            sizes = 20 + vol_norm * 40
            ax.scatter(cells[:, 0], cells[:, 1], cells[:, 2], c=colors, s=sizes, alpha=0.9)
            for u, v, d in edges:
                ax.plot([cells[u, 0], cells[v, 0]],
                       [cells[u, 1], cells[v, 1]],
                       [cells[u, 2], cells[v, 2]],
                       color='#FF9800', alpha=0.4, linewidth=0.8)
        else:  # Classification result
            sizes = 20 + vol_norm * 40
            ax.scatter(cells[:, 0], cells[:, 1], cells[:, 2], c='#9C27B0', s=sizes, alpha=0.9)
            for u, v, d in edges:
                ax.plot([cells[u, 0], cells[v, 0]],
                       [cells[u, 1], cells[v, 1]],
                       [cells[u, 2], cells[v, 2]],
                       color='#9C27B0', alpha=0.3, linewidth=0.8)
        
        # Add arrow between steps
        if i < len(steps) - 1:
            fig.text(0.18 + i * 0.2, 0.5, '→', fontsize=30, color='white', 
                    ha='center', va='center', fontweight='bold')
    
    plt.suptitle('Organoid Analysis Pipeline: From Image to Prediction', 
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('graph_construction_pipeline.png', dpi=DPI, 
                facecolor='#0d1117', bbox_inches='tight')
    print("Saved graph_construction_pipeline.png")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("Graph Construction Animation")
    print("=" * 60)
    
    # Generate main animation
    animate_graph_construction()
    
    # Generate static pipeline figure
    print("\nGenerating pipeline figure...")
    create_pipeline_figure()
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)
