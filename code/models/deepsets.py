"""
DeepSets: Deep Sets for Set-based Learning

Reference: Zaheer et al. (2017), Deep Sets
Application: Organoid classification via set of cells (permutation-invariant aggregation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DeepSets(nn.Module):
    """
    DeepSets architecture for permutation-invariant set processing
    
    Architecture: ρ(Σ φ(x_i)) where:
    - φ: encoder MLP applied to each element independently
    - Σ: permutation-invariant aggregation (sum, max, mean, or combination)
    - ρ: decoder MLP applied to aggregated representation
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        encoder_layers: int = 3,
        encoder_channels: list = None,
        out_channels: int = 64,
        dropout: float = 0.5,
        aggregation: str = 'sum_max',  # 'sum', 'max', 'mean', 'sum_max', 'sum_max_mean'
        batch_norm: bool = True,
    ):
        """
        Initialize DeepSets
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden units
            encoder_layers: Number of layers in encoder φ
            encoder_channels: Custom list of channel sizes for encoder (overrides hidden_channels)
            out_channels: Number of output channels (embedding dimension)
            dropout: Dropout rate
            aggregation: Aggregation method ('sum', 'max', 'mean', 'sum_max', 'sum_max_mean')
            batch_norm: Use batch normalization
        """
        super(DeepSets, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.encoder_layers = encoder_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.aggregation = aggregation
        self.use_batch_norm = batch_norm
        
        # Determine encoder architecture
        if encoder_channels is None:
            encoder_channels = [hidden_channels] * encoder_layers
        
        # Build encoder φ (applied to each element independently)
        encoder_modules = []
        prev_channels = in_channels
        
        for i, channels in enumerate(encoder_channels):
            encoder_modules.append(nn.Linear(prev_channels, channels))
            if batch_norm:
                encoder_modules.append(nn.BatchNorm1d(channels))
            encoder_modules.append(nn.ReLU())
            if dropout > 0:
                encoder_modules.append(nn.Dropout(dropout))
            prev_channels = channels
        
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Determine aggregated dimension
        if aggregation == 'sum' or aggregation == 'max' or aggregation == 'mean':
            self.agg_dim = prev_channels
        elif aggregation == 'sum_max':
            self.agg_dim = prev_channels * 2
        elif aggregation == 'sum_max_mean':
            self.agg_dim = prev_channels * 3
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Final projection to output dimension
        self.output_proj = nn.Sequential(
            nn.Linear(self.agg_dim, out_channels),
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        **kwargs  # Accept edge_index for compatibility but ignore it
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, in_channels)
            batch: Batch assignment (num_nodes,) for graph-level tasks
            **kwargs: Ignored (for compatibility with GNN interface)
        
        Returns:
            Graph embeddings (batch_size, out_channels) if batch provided,
            else global embedding (1, out_channels)
        """
        # Encode each element independently: φ(x_i)
        encoded = self.encoder(x)  # (num_nodes, encoder_output_dim)
        
        # Aggregate across set
        if batch is None:
            # Single set - aggregate all
            aggregated = self._aggregate(encoded.unsqueeze(0))  # (1, agg_dim)
        else:
            # Multiple sets - aggregate per batch
            aggregated = self._aggregate_batched(encoded, batch)  # (batch_size, agg_dim)
        
        # Decode: ρ(aggregated)
        output = self.output_proj(aggregated)
        
        return output
    
    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate across set dimension (dim=1)
        
        Args:
            x: Tensor of shape (batch_size, num_elements, channels)
        
        Returns:
            Aggregated tensor (batch_size, agg_dim)
        """
        if self.aggregation == 'sum':
            return x.sum(dim=1)
        elif self.aggregation == 'max':
            return x.max(dim=1)[0]
        elif self.aggregation == 'mean':
            return x.mean(dim=1)
        elif self.aggregation == 'sum_max':
            sum_agg = x.sum(dim=1)
            max_agg = x.max(dim=1)[0]
            return torch.cat([sum_agg, max_agg], dim=-1)
        elif self.aggregation == 'sum_max_mean':
            sum_agg = x.sum(dim=1)
            max_agg = x.max(dim=1)[0]
            mean_agg = x.mean(dim=1)
            return torch.cat([sum_agg, max_agg, mean_agg], dim=-1)
    
    def _aggregate_batched(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Aggregate within each batch
        
        Args:
            x: Node features (num_nodes, channels)
            batch: Batch indices (num_nodes,)
        
        Returns:
            Aggregated per batch (batch_size, agg_dim)
        """
        batch_size = batch.max().item() + 1
        channels = x.size(1)
        
        # Initialize aggregation tensors
        if self.aggregation == 'sum':
            result = torch.zeros(batch_size, channels, device=x.device, dtype=x.dtype)
            result.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        
        elif self.aggregation == 'max':
            result = torch.full((batch_size, channels), float('-inf'), device=x.device, dtype=x.dtype)
            for b in range(batch_size):
                mask = (batch == b)
                if mask.any():
                    result[b] = x[mask].max(dim=0)[0]
            result[result == float('-inf')] = 0  # Handle empty batches
        
        elif self.aggregation == 'mean':
            result = torch.zeros(batch_size, channels, device=x.device, dtype=x.dtype)
            counts = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            result.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
            counts.scatter_add_(0, batch.unsqueeze(1), torch.ones_like(batch, dtype=x.dtype).unsqueeze(1))
            result = result / counts.clamp(min=1)
        
        elif self.aggregation == 'sum_max':
            sum_result = torch.zeros(batch_size, channels, device=x.device, dtype=x.dtype)
            sum_result.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
            
            max_result = torch.full((batch_size, channels), float('-inf'), device=x.device, dtype=x.dtype)
            for b in range(batch_size):
                mask = (batch == b)
                if mask.any():
                    max_result[b] = x[mask].max(dim=0)[0]
            max_result[max_result == float('-inf')] = 0
            
            result = torch.cat([sum_result, max_result], dim=-1)
        
        elif self.aggregation == 'sum_max_mean':
            sum_result = torch.zeros(batch_size, channels, device=x.device, dtype=x.dtype)
            counts = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            sum_result.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
            counts.scatter_add_(0, batch.unsqueeze(1), torch.ones_like(batch, dtype=x.dtype).unsqueeze(1))
            mean_result = sum_result / counts.clamp(min=1)
            
            max_result = torch.full((batch_size, channels), float('-inf'), device=x.device, dtype=x.dtype)
            for b in range(batch_size):
                mask = (batch == b)
                if mask.any():
                    max_result[b] = x[mask].max(dim=0)[0]
            max_result[max_result == float('-inf')] = 0
            
            result = torch.cat([sum_result, max_result, mean_result], dim=-1)
        
        return result
    
    def get_graph_embedding(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Get graph-level embedding (for compatibility with GNN interface)
        
        Args:
            x: Node features
            batch: Batch assignment
            **kwargs: Ignored
        
        Returns:
            Graph embeddings (batch_size, out_channels)
        """
        return self.forward(x, batch, **kwargs)


class DeepSetsClassifier(nn.Module):
    """
    DeepSets with classification head for organoid classification
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.5,
        aggregation: str = 'sum_max',
    ):
        """
        Initialize DeepSets classifier
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden units
            num_layers: Number of encoder layers
            num_classes: Number of output classes
            dropout: Dropout rate
            aggregation: Aggregation method
        """
        super(DeepSetsClassifier, self).__init__()
        
        # Build encoder channel progression
        encoder_channels = []
        encoder_channels.append(hidden_channels)
        for _ in range(num_layers - 2):
            encoder_channels.append(hidden_channels * 2)
        encoder_channels.append(hidden_channels * 2)
        
        # DeepSets encoder
        self.encoder = DeepSets(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            encoder_layers=num_layers,
            encoder_channels=encoder_channels,
            out_channels=hidden_channels * 2,
            dropout=dropout,
            aggregation=aggregation,
        )
        
        # Classification head (decoder ρ)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,  # Ignored, for compatibility
    ) -> torch.Tensor:
        """
        Forward pass for classification
        
        Args:
            x: Node features (num_nodes, in_channels)
            batch: Batch assignment (num_nodes,)
            edge_index: Ignored (for compatibility with GNN interface)
        
        Returns:
            Logits (batch_size, num_classes)
        """
        # Get set embeddings
        set_emb = self.encoder(x, batch)
        
        # Classify
        logits = self.classifier(set_emb)
        
        return logits
    
    def predict(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get class predictions
        
        Args:
            x: Node features
            batch: Batch assignment
            edge_index: Ignored
        
        Returns:
            Predicted class labels (batch_size,)
        """
        logits = self.forward(x, batch, edge_index)
        preds = torch.argmax(logits, dim=1)
        return preds
    
    def predict_proba(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get class probabilities
        
        Args:
            x: Node features
            batch: Batch assignment
            edge_index: Ignored
        
        Returns:
            Class probabilities (batch_size, num_classes)
        """
        logits = self.forward(x, batch, edge_index)
        probs = F.softmax(logits, dim=1)
        return probs


if __name__ == "__main__":
    # Test DeepSets
    print("Testing DeepSets...")
    
    # Create dummy data
    num_nodes = 100
    in_channels = 4  # 3D position + volume
    batch_size = 8
    
    x = torch.randn(num_nodes, in_channels)
    batch = torch.randint(0, batch_size, (num_nodes,))
    
    # Model
    model = DeepSetsClassifier(
        in_channels=in_channels,
        hidden_channels=128,
        num_layers=3,
        num_classes=5,
        aggregation='sum_max',
    )
    
    # Forward pass
    output = model(x, batch)
    
    print(f"Input shape: {x.shape}")
    print(f"Batch shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test different aggregations
    print("\nTesting aggregation methods...")
    for agg in ['sum', 'max', 'mean', 'sum_max', 'sum_max_mean']:
        model_agg = DeepSetsClassifier(
            in_channels=in_channels,
            hidden_channels=64,
            num_layers=3,
            num_classes=2,
            aggregation=agg,
        )
        out = model_agg(x, batch)
        print(f"  {agg:15s}: output shape = {out.shape}, params = {sum(p.numel() for p in model_agg.parameters()):,}")
    
    print("\n✓ DeepSets tests passed!")

