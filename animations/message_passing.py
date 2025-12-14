"""
Animation: Message Passing in Graph Neural Networks
Shows how information propagates through a graph layer by layer.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import networkx as nx

# Configuration
np.random.seed(42)
FIG_SIZE = (10, 8)
DPI = 150
SAVE_GIF = True

def create_sample_graph():
    """Create a sample graph representing an organoid structure."""
    # Create a clustered graph
    G = nx.Graph()
    
    # Central node
    positions = {0: (0, 0)}
    
    # First ring
    for i in range(1, 7):
        angle = 2 * np.pi * (i - 1) / 6
        positions[i] = (1.5 * np.cos(angle), 1.5 * np.sin(angle))
    
    # Second ring
    for i in range(7, 19):
        angle = 2 * np.pi * (i - 7) / 12 + np.pi/12
        positions[i] = (2.8 * np.cos(angle), 2.8 * np.sin(angle))
    
    # Add nodes
    for i in positions:
        G.add_node(i, pos=positions[i])
    
    # Add edges (k-nearest neighbors style)
    for i in G.nodes():
        pos_i = np.array(positions[i])
        distances = []
        for j in G.nodes():
            if i != j:
                pos_j = np.array(positions[j])
                distances.append((j, np.linalg.norm(pos_i - pos_j)))
        distances.sort(key=lambda x: x[1])
        for j, d in distances[:4]:  # Connect to 4 nearest neighbors
            G.add_edge(i, j)
    
    return G, positions


def animate_message_passing():
    """Create animation showing message passing layers."""
    G, positions = create_sample_graph()
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Color scheme
    colors_by_layer = {
        0: '#4CAF50',  # Green - initial
        1: '#2196F3',  # Blue - layer 1
        2: '#9C27B0',  # Purple - layer 2
        3: '#FF9800',  # Orange - layer 3
    }
    
    # Draw edges
    edge_lines = []
    for u, v in G.edges():
        x = [positions[u][0], positions[v][0]]
        y = [positions[u][1], positions[v][1]]
        line, = ax.plot(x, y, color='#ffffff', alpha=0.3, linewidth=1.5, zorder=1)
        edge_lines.append(line)
    
    # Draw nodes
    node_circles = {}
    node_texts = {}
    for node in G.nodes():
        x, y = positions[node]
        circle = Circle((x, y), 0.25, color=colors_by_layer[0], zorder=2)
        ax.add_patch(circle)
        node_circles[node] = circle
        text = ax.text(x, y, str(node), ha='center', va='center', 
                      fontsize=8, fontweight='bold', color='white', zorder=3)
        node_texts[node] = text
    
    # Title and layer indicator
    title = ax.text(0, 3.5, 'Message Passing: Layer 0 (Initial Features)', 
                   ha='center', fontsize=14, fontweight='bold', color='white')
    
    # Legend
    for i, (layer, color) in enumerate(colors_by_layer.items()):
        ax.add_patch(Circle((-3.5, 3 - i*0.5), 0.15, color=color))
        ax.text(-3.2, 3 - i*0.5, f'Layer {layer}', va='center', fontsize=10, color='white')
    
    # Animation state
    current_layer = [0]
    active_nodes = [set([0])]  # Start from center node
    message_arrows = []
    
    def get_neighbors_at_distance(node, distance):
        """Get nodes at exact distance from source."""
        if distance == 0:
            return {node}
        visited = {node}
        current = {node}
        for _ in range(distance):
            next_level = set()
            for n in current:
                for neighbor in G.neighbors(n):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
            current = next_level
        return current
    
    def init():
        return list(node_circles.values()) + edge_lines + [title]
    
    def animate(frame):
        # Clear previous arrows
        for arrow in message_arrows:
            arrow.remove()
        message_arrows.clear()
        
        layer = frame // 30  # 30 frames per layer
        sub_frame = frame % 30
        
        if layer > 3:
            layer = 3
            sub_frame = 29
        
        # Update title
        if sub_frame < 15:
            title.set_text(f'Message Passing: Layer {layer} - Aggregating Messages')
        else:
            title.set_text(f'Message Passing: Layer {layer} - Updated Features')
        
        # Get nodes that should be active at this layer
        active = get_neighbors_at_distance(0, layer)
        
        # Animate message passing
        if sub_frame < 15:
            # Show messages flowing
            progress = sub_frame / 15
            for node in active:
                for neighbor in G.neighbors(node):
                    if neighbor in get_neighbors_at_distance(0, layer - 1) if layer > 0 else {0}:
                        # Draw arrow from neighbor to node
                        x1, y1 = positions[neighbor]
                        x2, y2 = positions[node]
                        
                        # Interpolate position
                        xi = x1 + (x2 - x1) * progress
                        yi = y1 + (y2 - y1) * progress
                        
                        arrow = ax.annotate('', xy=(xi, yi), xytext=(x1, y1),
                                          arrowprops=dict(arrowstyle='->', color='#FF5722', 
                                                         lw=2, alpha=0.8))
                        message_arrows.append(arrow)
        
        # Update node colors
        for node in G.nodes():
            dist = nx.shortest_path_length(G, 0, node)
            if dist <= layer:
                if sub_frame >= 15 or dist < layer:
                    node_circles[node].set_color(colors_by_layer[min(layer, 3)])
                    node_circles[node].set_alpha(1.0)
                else:
                    # Pulsing effect during message aggregation
                    pulse = 0.7 + 0.3 * np.sin(sub_frame * np.pi / 7.5)
                    node_circles[node].set_alpha(pulse)
            else:
                node_circles[node].set_color(colors_by_layer[0])
                node_circles[node].set_alpha(0.5)
        
        return list(node_circles.values()) + edge_lines + [title] + message_arrows
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=120, interval=50, blit=False)
    
    if SAVE_GIF:
        print("Saving message_passing.gif...")
        anim.save('animations/message_passing.gif', writer='pillow', fps=20, dpi=DPI)
        print("Saved!")
    
    plt.show()
    return anim


if __name__ == "__main__":
    import os
    os.makedirs('animations', exist_ok=True)
    animate_message_passing()
