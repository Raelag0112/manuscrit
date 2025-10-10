"""
Graph Isomorphism Network (GIN)

Reference: Xu et al. (2019), How Powerful are Graph Neural Networks?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool


class GIN(nn.Module):
    """Graph Isomorphism Network"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 5,
        out_channels: int = 64,
        dropout: float = 0.5,
        train_eps: bool = True,
    ):
        super(GIN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp, train_eps=train_eps))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.convs.append(GINConv(mlp, train_eps=train_eps))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def get_graph_embedding(self, x, edge_index, batch):
        node_embeddings = self.forward(x, edge_index)
        # GIN paper uses sum pooling
        return global_add_pool(node_embeddings, batch)


class GINClassifier(nn.Module):
    """GIN with classification head"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 5,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super(GINClassifier, self).__init__()
        
        self.encoder = GIN(
            in_channels, hidden_channels, num_layers,
            hidden_channels, dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch):
        graph_emb = self.encoder.get_graph_embedding(x, edge_index, batch)
        return self.classifier(graph_emb)
    
    def predict(self, x, edge_index, batch):
        logits = self.forward(x, edge_index, batch)
        return torch.argmax(logits, dim=1)

