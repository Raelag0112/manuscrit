"""
Unified organoid classifier interface
"""

import torch
import torch.nn as nn
from .gcn import GCNClassifier
from .gat import GATClassifier
from .graphsage import GraphSAGEClassifier
from .gin import GINClassifier
from .egnn import EGNNClassifier


class OrganoidClassifier:
    """
    Factory for creating organoid classifiers with different GNN architectures
    """
    
    MODELS = {
        'gcn': GCNClassifier,
        'gat': GATClassifier,
        'graphsage': GraphSAGEClassifier,
        'gin': GINClassifier,
        'egnn': EGNNClassifier,
    }
    
    @staticmethod
    def create(
        model_type: str,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.5,
        **kwargs
    ) -> nn.Module:
        """
        Create organoid classifier
        
        Args:
            model_type: Type of GNN ('gcn', 'gat', 'graphsage', 'gin', 'egnn')
            in_channels: Number of input node features
            num_classes: Number of output classes
            hidden_channels: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            **kwargs: Additional model-specific arguments
        
        Returns:
            Classifier model
        """
        if model_type not in OrganoidClassifier.MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(OrganoidClassifier.MODELS.keys())}"
            )
        
        model_class = OrganoidClassifier.MODELS[model_type]
        
        model = model_class(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            **kwargs
        )
        
        return model
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path: str, device='cpu') -> nn.Module:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: Device to load model on
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model configuration
        model_config = checkpoint.get('model_config', {})
        model_type = model_config.get('model_type', 'gcn')
        
        # Create model
        model = OrganoidClassifier.create(**model_config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model


if __name__ == "__main__":
    # Test model creation
    for model_type in ['gcn', 'gat', 'graphsage', 'gin', 'egnn']:
        print(f"\nCreating {model_type.upper()} model...")
        model = OrganoidClassifier.create(
            model_type=model_type,
            in_channels=10,
            num_classes=5,
            hidden_channels=64,
            num_layers=3,
        )
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

