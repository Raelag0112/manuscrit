"""
Animation: Graph Attention Mechanism (GAT)
Visualizes how attention weights are computed and used for aggregation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
import matplotlib.colors as mcolors

# Configuration
np.random.seed(42)
FIG_SIZE = (12, 8)
DPI = 150
SAVE_GIF = True


def create_local_graph():
    """Create a small graph to illustrate attention."""
    # Central node and its neighbors
    center = (0, 0)
    neighbors = [
        (-1.5, 1.2),   # Top-left
        (0, 1.8),      # Top
        (1.5, 1.2),    # Top-right
        (1.8, -0.3),   # Right
        (0.8, -1.5),   # Bottom-right
        (-0.8, -1.5),  # Bottom-left
        (-1.8, -0.3),  # Left
    ]
    
    # Simulated attention weights (should sum to 1)
    # Higher weights for certain neighbors to show differentiation
    attention_weights = [0.25, 0.15, 0.20, 0.08, 0.12, 0.10, 0.10]
    
    return center, neighbors, attention_weights


def animate_attention():
    """Create animation showing attention mechanism."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    center, neighbors, attention_weights = create_local_graph()
    
    # Title
    title = ax.text(0, 2.7, 'Graph Attention Network (GAT)', 
                   ha='center', fontsize=16, fontweight='bold', color='white')
    subtitle = ax.text(0, 2.4, 'Computing attention for node i', 
                      ha='center', fontsize=12, color='#888888')
    
    # Color gradient for attention
    cmap = plt.cm.YlOrRd
    
    # Draw edges (initially hidden)
    edge_lines = []
    edge_arrows = []
    for j, (nx, ny) in enumerate(neighbors):
        # Line
        line, = ax.plot([center[0], nx], [center[1], ny], 
                       color='#444444', linewidth=2, alpha=0, zorder=1)
        edge_lines.append(line)
    
    # Draw neighbor nodes
    neighbor_circles = []
    neighbor_texts = []
    for j, (nx, ny) in enumerate(neighbors):
        circle = Circle((nx, ny), 0.25, color='#2196F3', zorder=2)
        ax.add_patch(circle)
        neighbor_circles.append(circle)
        
        text = ax.text(nx, ny, f'j{j+1}', ha='center', va='center', 
                      fontsize=10, fontweight='bold', color='white', zorder=3)
        neighbor_texts.append(text)
    
    # Draw central node
    center_circle = Circle(center, 0.3, color='#4CAF50', zorder=4)
    ax.add_patch(center_circle)
    center_text = ax.text(center[0], center[1], 'i', ha='center', va='center', 
                         fontsize=12, fontweight='bold', color='white', zorder=5)
    
    # Attention weight labels
    weight_texts = []
    for j, (nx, ny) in enumerate(neighbors):
        mx, my = (center[0] + nx) / 2, (center[1] + ny) / 2
        text = ax.text(mx, my + 0.2, '', ha='center', va='center', 
                      fontsize=9, color='white', fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
        weight_texts.append(text)
    
    # Formula display
    formula_box = ax.text(-2.5, -2.2, '', fontsize=10, color='white',
                         family='monospace', verticalalignment='top')
    
    # Legend
    ax.text(2.2, -1.8, 'Attention Weight', fontsize=10, color='white')
    for i, val in enumerate([0.0, 0.5, 1.0]):
        y = -2.0 - i * 0.25
        color = cmap(val)
        ax.add_patch(Circle((2.3, y), 0.1, color=color))
        ax.text(2.5, y, f'{val:.1f}', va='center', fontsize=9, color='white')
    
    def init():
        return [center_circle, center_text, title, subtitle, formula_box] + \
               neighbor_circles + neighbor_texts + edge_lines + weight_texts
    
    def animate(frame):
        phase = frame // 40
        sub_frame = frame % 40
        
        # Phase 0: Show structure (frames 0-39)
        if phase == 0:
            subtitle.set_text('Step 1: Identify neighbors of node i')
            formula_box.set_text('')
            
            # Fade in edges
            alpha = min(1, sub_frame / 20)
            for line in edge_lines:
                line.set_alpha(alpha * 0.5)
        
        # Phase 1: Compute attention scores (frames 40-79)
        elif phase == 1:
            subtitle.set_text('Step 2: Compute attention scores')
            formula_box.set_text('eᵢⱼ = LeakyReLU(aᵀ[Whᵢ ∥ Whⱼ])')
            
            # Highlight edges one by one
            current_edge = sub_frame // 6
            for j, line in enumerate(edge_lines):
                if j < current_edge:
                    line.set_color('#FF9800')
                    line.set_alpha(0.8)
                    line.set_linewidth(3)
                elif j == current_edge:
                    # Pulsing current edge
                    pulse = 0.5 + 0.5 * np.sin(sub_frame * 0.5)
                    line.set_color('#FF9800')
                    line.set_alpha(pulse)
                    line.set_linewidth(4)
                else:
                    line.set_color('#444444')
                    line.set_alpha(0.5)
                    line.set_linewidth(2)
        
        # Phase 2: Softmax normalization (frames 80-119)
        elif phase == 2:
            subtitle.set_text('Step 3: Normalize with softmax')
            formula_box.set_text('αᵢⱼ = softmax(eᵢⱼ) = exp(eᵢⱼ) / Σₖ exp(eᵢₖ)')
            
            # Show weights appearing
            progress = sub_frame / 40
            for j, (text, weight) in enumerate(zip(weight_texts, attention_weights)):
                if progress > j / len(neighbors):
                    text.set_text(f'α={weight:.2f}')
                    
                    # Color edge by attention weight
                    color = cmap(weight / max(attention_weights))
                    edge_lines[j].set_color(color)
                    edge_lines[j].set_linewidth(2 + weight * 8)
                    edge_lines[j].set_alpha(0.8)
        
        # Phase 3: Weighted aggregation (frames 120-159)
        elif phase == 3:
            subtitle.set_text('Step 4: Aggregate neighbor features')
            formula_box.set_text("h'ᵢ = σ(Σⱼ αᵢⱼ · Whⱼ)")
            
            # Animate messages flowing to center
            progress = sub_frame / 40
            for j, ((nx, ny), weight) in enumerate(zip(neighbors, attention_weights)):
                # Message position
                mx = nx + (center[0] - nx) * progress
                my = ny + (center[1] - ny) * progress
                
                # Update edge to show flow
                edge_lines[j].set_data([nx, mx], [ny, my])
            
            # Center node grows based on aggregation
            scale = 0.3 + 0.1 * progress
            center_circle.set_radius(scale)
            
            # Pulse center color
            if progress > 0.8:
                pulse = (progress - 0.8) / 0.2
                r, g, b = 0.3, 0.69, 0.31  # Green
                r2, g2, b2 = 1.0, 0.6, 0.0  # Orange
                new_color = (r + (r2-r)*pulse, g + (g2-g)*pulse, b + (b2-b)*pulse)
                center_circle.set_color(new_color)
        
        # Phase 4: Final state (frames 160+)
        else:
            subtitle.set_text('Result: Updated node representation h\'ᵢ')
            formula_box.set_text("h'ᵢ incorporates weighted information from all neighbors")
            
            # Reset edges
            for j, ((nx, ny), weight) in enumerate(zip(neighbors, attention_weights)):
                edge_lines[j].set_data([center[0], nx], [center[1], ny])
                color = cmap(weight / max(attention_weights))
                edge_lines[j].set_color(color)
            
            # Pulse center
            pulse = 0.9 + 0.1 * np.sin(sub_frame * 0.3)
            center_circle.set_radius(0.4 * pulse)
            center_circle.set_color('#FF9800')
        
        return [center_circle, center_text, title, subtitle, formula_box] + \
               neighbor_circles + neighbor_texts + edge_lines + weight_texts
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=200, interval=50, blit=False)
    
    if SAVE_GIF:
        print("Saving attention_mechanism.gif...")
        anim.save('animations/attention_mechanism.gif', writer='pillow', fps=20, dpi=DPI)
        print("Saved!")
    
    plt.show()
    return anim


if __name__ == "__main__":
    import os
    os.makedirs('animations', exist_ok=True)
    animate_attention()
