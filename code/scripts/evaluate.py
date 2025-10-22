"""
Evaluation script for trained models
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import logging
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.classifier import OrganoidClassifier
from data.loader import get_dataloaders
from utils.metrics import compute_metrics, OrganoidMetrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device to use
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    model.to(device)
    
    metrics_tracker = OrganoidMetrics()
    
    for batch in test_loader:
        batch = batch.to(device)
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        probs = F.softmax(out, dim=1)
        
        # Update metrics
        metrics_tracker.update(
            pred.cpu().numpy(),
            batch.y.cpu().numpy(),
            probs.cpu().numpy()
        )
    
    # Compute final metrics
    metrics = metrics_tracker.compute()
    conf_matrix = metrics_tracker.get_confusion_matrix()
    class_report = metrics_tracker.get_classification_report()
    
    return metrics, conf_matrix, class_report


def main():
    parser = argparse.ArgumentParser(description='Evaluate organoid classifier')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory with test split')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    
    # Create model (you'll need to save model config in checkpoint)
    # For now, assume we can infer from checkpoint
    model = OrganoidClassifier.load_from_checkpoint(args.model_path, args.device)
    
    # Load test data
    logger.info(f"Loading test data from {args.data_dir}")
    _, _, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics, conf_matrix, class_report = evaluate_model(
        model, test_loader, args.device
    )
    
    # Log results
    logger.info("=" * 50)
    logger.info("Test Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(str(conf_matrix))
    
    logger.info("\nClassification Report:")
    logger.info(class_report)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    np.save(output_dir / 'confusion_matrix.npy', conf_matrix)
    
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(class_report)
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

