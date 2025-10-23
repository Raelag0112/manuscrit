"""
Training script for organoid classification with GNNs
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
import argparse
import logging
import json
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.classifier import OrganoidClassifier
from data.loader import get_dataloaders
from utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for organoid classification
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=0.001,
        weight_decay=1e-4,
        scheduler_type='plateau',
        model_config=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_config = model_config or {}
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        # Training state
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            batch = batch.to(self.device)
            
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(all_labels)
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def train(self, num_epochs, save_dir):
        """
        Train model
        
        Args:
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            val_acc = val_metrics['accuracy']
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Log
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            
            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(save_dir / 'best_model.pth', epoch, val_metrics)
                logger.info(f"Saved new best model with val acc: {val_acc:.4f}")
            
            # Save latest
            self.save_checkpoint(save_dir / 'latest_model.pth', epoch, val_metrics)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
        }
        
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training complete! Best val acc: {self.best_val_acc:.4f}")
    
    def save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'model_config': self.model_config,
        }
        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description='Train organoid classifier')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory with train/val/test splits')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    
    # Model
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gat', 'graphsage', 'gin', 'egnn', 'deepsets'],
                        help='Model architecture (GNN or DeepSets baseline)')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='LR scheduler')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    logger.info(f"Train: {len(train_loader.dataset)} samples")
    logger.info(f"Val: {len(val_loader.dataset)} samples")
    logger.info(f"Test: {len(test_loader.dataset)} samples")
    
    # Get data info
    sample_batch = next(iter(train_loader))
    in_channels = sample_batch.x.size(1)
    num_classes = len(sample_batch.y.unique())
    
    logger.info(f"Input features: {in_channels}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    logger.info(f"Creating {args.model.upper()} model")
    model_config = {
        'model_type': args.model,
        'in_channels': in_channels,
        'num_classes': num_classes,
        'hidden_channels': args.hidden_channels,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
    }
    model = OrganoidClassifier.create(**model_config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        model_config=model_config,
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        save_dir=args.output_dir,
    )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

