"""
Interpretability tools for GNN predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data


class GNNExplainer:
    """
    Simple GNN explainer using gradient-based attribution
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def explain(
        self,
        data: Data,
        target_class: int = None,
    ):
        """
        Compute node importance scores
        
        Args:
            data: Input graph
            target_class: Class to explain (None = predicted class)
        
        Returns:
            node_scores: Importance score for each node
        """
        data = data.to(self.device)
        data.x.requires_grad = True
        
        # Forward pass
        out = self.model(data.x, data.edge_index, data.batch)
        
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        out[0, target_class].backward()
        
        # Get gradients as importance scores
        node_scores = data.x.grad.abs().sum(dim=1).cpu().numpy()
        
        return node_scores, target_class
    
    def visualize_explanation(
        self,
        data: Data,
        node_scores: np.ndarray,
        output_path: str = None,
    ):
        """
        Visualize node importance
        
        Args:
            data: Input graph
            node_scores: Node importance scores
            output_path: Path to save figure
        """
        fig = plt.figure(figsize=(12, 5))
        
        pos = data.pos.cpu().numpy()
        
        # Normalize scores for color mapping
        scores_norm = (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min() + 1e-8)
        
        # XY projection
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(
            pos[:, 0], pos[:, 1],
            c=scores_norm,
            cmap='Reds',
            s=100,
            alpha=0.7
        )
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Node Importance - XY projection')
        ax1.axis('equal')
        plt.colorbar(scatter, ax=ax1)
        
        # XZ projection
        ax2 = fig.add_subplot(122)
        scatter = ax2.scatter(
            pos[:, 0], pos[:, 2],
            c=scores_norm,
            cmap='Reds',
            s=100,
            alpha=0.7
        )
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('Node Importance - XZ projection')
        ax2.axis('equal')
        plt.colorbar(scatter, ax=ax2)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def visualize_attention(
    data: Data,
    attention_weights: list,
    layer_idx: int = 0,
    output_path: str = None,
):
    """
    Visualize GAT attention weights
    
    Args:
        data: Input graph
        attention_weights: List of attention weights per layer
        layer_idx: Which layer to visualize
        output_path: Path to save figure
    """
    edge_index, alpha = attention_weights[layer_idx]
    
    # Average attention over heads if multiple
    if alpha.dim() == 2:
        alpha = alpha.mean(dim=1)
    
    alpha = alpha.cpu().numpy()
    
    # Normalize
    alpha_norm = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    pos = data.pos.cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    
    # Plot edges with attention-based thickness/color
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        alpha_val = alpha_norm[i]
        
        plt.plot(
            [pos[src, 0], pos[dst, 0]],
            [pos[src, 1], pos[dst, 1]],
            color=plt.cm.Reds(alpha_val),
            alpha=alpha_val,
            linewidth=1 + 2 * alpha_val,
        )
    
    # Plot nodes
    plt.scatter(pos[:, 0], pos[:, 1], c='steelblue', s=100, zorder=10)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Attention Weights - Layer {layer_idx}')
    plt.axis('equal')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("GNN interpretability tools")
    print("Use with trained models for attribution analysis")

