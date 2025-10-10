"""
Graph Convolutional Network (GCN)

Reference: Kipf & Welling (2017), Semi-Supervised Classification with Graph Convolutional Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from typing import Optional


class GCN(nn.Module):
    """
    Graph Convolutional Network for organoid classification
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        out_channels: int = 64,
        dropout: float = 0.5,
        batch_norm: bool = True,
    ):
        """
        Initialize GCN
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden units
            num_layers: Number of GCN layers
            out_channels: Number of output channels (embedding dimension)
            dropout: Dropout rate
            batch_norm: Use batch normalization
        """
        super(GCN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_batch_norm = batch_norm
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        # Batch normalization
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            batch: Batch assignment (num_nodes,) for graph-level tasks
        
        Returns:
            Node embeddings (num_nodes, out_channels)
        """
        # Apply GCN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)
        if self.use_batch_norm:
            x = self.batch_norms[-1](x)
        
        return x
    
    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        pooling: str = 'mean',
    ) -> torch.Tensor:
        """
        Get graph-level embedding
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            pooling: Pooling method ('mean', 'max', 'add')
        
        Returns:
            Graph embeddings (batch_size, out_channels)
        """
        # Get node embeddings
        node_embeddings = self.forward(x, edge_index, batch)
        
        # Pool to graph level
        if pooling == 'mean':
            graph_embedding = global_mean_pool(node_embeddings, batch)
        elif pooling == 'max':
            graph_embedding = global_max_pool(node_embeddings, batch)
        elif pooling == 'add':
            graph_embedding = global_add_pool(node_embeddings, batch)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        return graph_embedding


class GCNClassifier(nn.Module):
    """
    GCN with classification head for organoid classification
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.5,
        pooling: str = 'mean',
    ):
        """
        Initialize GCN classifier
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden units
            num_layers: Number of GCN layers
            num_classes: Number of output classes
            dropout: Dropout rate
            pooling: Graph pooling method
        """
        super(GCNClassifier, self).__init__()
        
        self.pooling = pooling
        
        # GCN encoder
        self.encoder = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for classification
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
        
        Returns:
            Logits (batch_size, num_classes)
        """
        # Get graph embeddings
        graph_emb = self.encoder.get_graph_embedding(
            x, edge_index, batch, pooling=self.pooling
        )
        
        # Classify
        logits = self.classifier(graph_emb)
        
        return logits
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get class predictions
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
        
        Returns:
            Predicted class labels (batch_size,)
        """
        logits = self.forward(x, edge_index, batch)
        preds = torch.argmax(logits, dim=1)
        return preds
    
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get class probabilities
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
        
        Returns:
            Class probabilities (batch_size, num_classes)
        """
        logits = self.forward(x, edge_index, batch)
        probs = F.softmax(logits, dim=1)
        return probs


if __name__ == "__main__":
    # Test GCN
    from torch_geometric.data import Data, Batch
    
    # Create dummy data
    num_nodes = 100
    in_channels = 10
    num_edges = 300
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Single graph
    data = Data(x=x, edge_index=edge_index)
    
    # Model
    model = GCNClassifier(
        in_channels=in_channels,
        hidden_channels=64,
        num_layers=3,
        num_classes=5,
    )
    
    # Forward pass
    batch = torch.zeros(num_nodes, dtype=torch.long)
    output = model(x, edge_index, batch)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

