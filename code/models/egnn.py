"""
E(n) Equivariant Graph Neural Network

Reference: Satorras et al. (2021), E(n) Equivariant Graph Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class EGNN_Layer(nn.Module):
    """Single E(n)-equivariant layer"""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 0):
        super(EGNN_Layer, self).__init__()
        
        # Edge MLP: computes messages using positions and features
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),  # +1 for distance^2
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Node MLP: updates node features
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Coordinate MLP: updates positions
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),  # No bias for E(n) equivariance
        )
    
    def forward(self, h, x, edge_index, edge_attr=None):
        """
        Args:
            h: Node features (N, hidden_dim)
            x: Node positions (N, 3)
            edge_index: Edge indices (2, E)
            edge_attr: Optional edge attributes (E, edge_dim)
        
        Returns:
            Updated h and x
        """
        row, col = edge_index
        
        # Compute relative positions and squared distances
        rel_pos = x[row] - x[col]  # (E, 3)
        dist_sq = torch.sum(rel_pos ** 2, dim=1, keepdim=True)  # (E, 1)
        
        # Edge features
        edge_feat = torch.cat([h[row], h[col], dist_sq], dim=1)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)
        
        # Compute edge messages
        edge_messages = self.edge_mlp(edge_feat)  # (E, hidden_dim)
        
        # Aggregate messages for each node
        messages = torch.zeros(h.size(0), edge_messages.size(1), device=h.device)
        messages.index_add_(0, col, edge_messages)
        
        # Update node features
        h_input = torch.cat([h, messages], dim=1)
        h = h + self.node_mlp(h_input)
        
        # Update coordinates
        coord_weights = self.coord_mlp(edge_messages)  # (E, 1)
        coord_updates = rel_pos * coord_weights  # (E, 3)
        
        coord_deltas = torch.zeros_like(x)
        coord_deltas.index_add_(0, col, coord_updates)
        x = x + coord_deltas
        
        return h, x


class EGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 5,
        out_channels: int = 128,
        edge_dim: int = 1,
    ):
        super(EGNN, self).__init__()
        
        # Input embedding
        self.node_embed = nn.Linear(in_channels, hidden_channels)
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNN_Layer(hidden_channels, edge_dim)
            for _ in range(num_layers)
        ])
        
        # Output
        self.output = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, h, pos, edge_index, edge_attr=None):
        """
        Args:
            h: Node features (N, in_channels)
            pos: Node positions (N, 3)
            edge_index: Edge indices (2, E)
            edge_attr: Edge attributes (E, edge_dim)
        
        Returns:
            Node embeddings and updated positions
        """
        # Embed node features
        h = self.node_embed(h)
        x = pos.clone()
        
        # Apply EGNN layers
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        
        # Output
        h = self.output(h)
        
        return h, x
    
    def get_graph_embedding(self, h, pos, edge_index, batch, edge_attr=None):
        node_embeddings, _ = self.forward(h, pos, edge_index, edge_attr)
        return global_mean_pool(node_embeddings, batch)


class EGNNClassifier(nn.Module):
    """EGNN with classification head
    
    Note: This implementation uses the first 3 features as positions if available,
    or creates dummy positions if data has no explicit 3D coordinates.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 5,
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super(EGNNClassifier, self).__init__()
        
        # If in_channels >= 3, use first 3 as positions, rest as features
        # Otherwise, create a position encoder
        self.use_position_encoding = in_channels < 3
        
        if self.use_position_encoding:
            # Create learnable position embeddings
            self.pos_encoder = nn.Linear(in_channels, 3)
            feature_dim = in_channels
        else:
            # Use first 3 features as positions, rest as node features
            feature_dim = max(in_channels - 3, 1)  # At least 1 feature
            if in_channels == 3:
                # If only 3 features, use them both as pos and features
                feature_dim = 3
        
        self.encoder = EGNN(
            feature_dim, hidden_channels, num_layers, hidden_channels, edge_dim=0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        Unified interface compatible with other GNN models
        
        Args:
            x: Node features (N, in_channels)
            edge_index: Edge indices (2, E)
            batch: Batch assignment (N,)
            edge_attr: Edge attributes (optional)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # Extract or create positions
        if self.use_position_encoding:
            # Create positions from features
            pos = self.pos_encoder(x)
            h = x  # Use all features
        else:
            # Split features into positions and node features
            if x.size(1) == 3:
                # Only 3 features: use them as both pos and features
                pos = x
                h = x
            else:
                # Split: first 3 as positions, rest as features
                pos = x[:, :3]
                h = x[:, 3:]
        
        # Get graph embedding
        graph_emb = self.encoder.get_graph_embedding(
            h, pos, edge_index, batch, edge_attr
        )
        return self.classifier(graph_emb)
    
    def predict(self, x, edge_index, batch, edge_attr=None):
        """Prediction with unified interface"""
        logits = self.forward(x, edge_index, batch, edge_attr)
        return torch.argmax(logits, dim=1)

