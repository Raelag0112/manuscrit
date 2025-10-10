"""
Graph Attention Network (GAT)

Reference: Veličković et al. (2018), Graph Attention Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Optional


class GAT(nn.Module):
    """
    Graph Attention Network for organoid analysis
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        out_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.6,
        attention_dropout: float = 0.6,
        concat_heads: bool = True,
    ):
        """
        Initialize GAT
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden units per head
            num_layers: Number of GAT layers
            out_channels: Number of output channels
            num_heads: Number of attention heads
            dropout: Feature dropout rate
            attention_dropout: Attention coefficient dropout
            concat_heads: Concatenate attention heads (True) or average (False)
        """
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=num_heads,
                dropout=attention_dropout,
                concat=concat_heads,
            )
        )
        self.batch_norms.append(
            nn.BatchNorm1d(hidden_channels * num_heads if concat_heads else hidden_channels)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            in_dim = hidden_channels * num_heads if concat_heads else hidden_channels
            self.convs.append(
                GATConv(
                    in_dim,
                    hidden_channels,
                    heads=num_heads,
                    dropout=attention_dropout,
                    concat=concat_heads,
                )
            )
            self.batch_norms.append(
                nn.BatchNorm1d(hidden_channels * num_heads if concat_heads else hidden_channels)
            )
        
        # Output layer (average heads)
        in_dim = hidden_channels * num_heads if concat_heads else hidden_channels
        self.convs.append(
            GATConv(
                in_dim,
                out_channels,
                heads=1,
                dropout=attention_dropout,
                concat=False,
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            return_attention_weights: Return attention weights
        
        Returns:
            Node embeddings (num_nodes, out_channels)
            Optionally: attention weights
        """
        attention_weights = []
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs[:-1]):
            if return_attention_weights:
                x, attn = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append(attn)
            else:
                x = conv(x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        if return_attention_weights:
            x, attn = self.convs[-1](x, edge_index, return_attention_weights=True)
            attention_weights.append(attn)
        else:
            x = self.convs[-1](x, edge_index)
        
        x = self.batch_norms[-1](x)
        
        if return_attention_weights:
            return x, attention_weights
        return x
    
    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get graph-level embedding
        """
        node_embeddings = self.forward(x, edge_index)
        graph_embedding = global_mean_pool(node_embeddings, batch)
        return graph_embedding


class GATClassifier(nn.Module):
    """
    GAT with classification head
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.6,
    ):
        super(GATClassifier, self).__init__()
        
        self.encoder = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
        )
        
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
        return_attention: bool = False,
    ):
        """
        Forward pass with optional attention weights
        """
        if return_attention:
            node_emb, attention_weights = self.encoder.forward(
                x, edge_index, return_attention_weights=True
            )
            graph_emb = global_mean_pool(node_emb, batch)
            logits = self.classifier(graph_emb)
            return logits, attention_weights
        else:
            graph_emb = self.encoder.get_graph_embedding(x, edge_index, batch)
            logits = self.classifier(graph_emb)
            return logits
    
    def predict(self, x, edge_index, batch):
        logits = self.forward(x, edge_index, batch)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x, edge_index, batch):
        logits = self.forward(x, edge_index, batch)
        return F.softmax(logits, dim=1)


if __name__ == "__main__":
    # Test GAT
    num_nodes = 100
    in_channels = 10
    edge_index = torch.randint(0, num_nodes, (2, 300))
    
    x = torch.randn(num_nodes, in_channels)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    model = GATClassifier(in_channels=in_channels, num_classes=5)
    
    output = model(x, edge_index, batch)
    print(f"Output shape: {output.shape}")
    
    # Test with attention
    output, attention = model(x, edge_index, batch, return_attention=True)
    print(f"Attention weights: {len(attention)} layers")

