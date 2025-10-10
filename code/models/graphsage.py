"""
GraphSAGE: Inductive Representation Learning on Large Graphs

Reference: Hamilton et al. (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGE(nn.Module):
    """GraphSAGE model"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        out_channels: int = 64,
        dropout: float = 0.5,
        aggr: str = 'mean',
    ):
        super(GraphSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        return x
    
    def get_graph_embedding(self, x, edge_index, batch):
        node_embeddings = self.forward(x, edge_index)
        return global_mean_pool(node_embeddings, batch)


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE with classification head"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super(GraphSAGEClassifier, self).__init__()
        
        self.encoder = GraphSAGE(
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

