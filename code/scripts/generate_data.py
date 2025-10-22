"""
Generate synthetic organoid dataset
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from synthetic.generator import SyntheticOrganoidGenerator

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic organoid dataset')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for generated data')
    parser.add_argument('--num_train', type=int, default=70000,
                        help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=15000,
                        help='Number of validation samples')
    parser.add_argument('--num_test', type=int, default=15000,
                        help='Number of test samples')
    parser.add_argument('--num_cells_min', type=int, default=50,
                        help='Minimum number of cells per organoid')
    parser.add_argument('--num_cells_max', type=int, default=500,
                        help='Maximum number of cells per organoid')
    parser.add_argument('--feature_dim', type=int, default=10,
                        help='Feature dimension')
    parser.add_argument('--radius', type=float, default=100.0,
                        help='Organoid radius')
    parser.add_argument('--edge_method', type=str, default='knn',
                        choices=['knn', 'radius', 'delaunay'],
                        help='Graph edge construction method')
    parser.add_argument('--k_neighbors', type=int, default=10,
                        help='Number of neighbors for KNN')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("SYNTHETIC ORGANOID GENERATION")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Dataset sizes: train={args.num_train}, val={args.num_val}, test={args.num_test}")
    logger.info(f"Cells per organoid: {args.num_cells_min}-{args.num_cells_max}")
    logger.info(f"Feature dimension: {args.feature_dim}")
    logger.info("=" * 60)
    
    # Create generator
    generator = SyntheticOrganoidGenerator(
        radius=args.radius,
        edge_method=args.edge_method,
        k_neighbors=args.k_neighbors,
        seed=args.seed,
    )
    
    # Generate dataset
    generator.generate_and_save(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        num_cells_range=(args.num_cells_min, args.num_cells_max),
        feature_dim=args.feature_dim,
    )
    
    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Total organoids: {args.num_train + args.num_val + args.num_test}")
    logger.info(f"Data saved to: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info(f"  1. Train a model:")
    logger.info(f"     python scripts/train.py --data_dir {args.output_dir} --output_dir results/")
    logger.info(f"  2. Evaluate:")
    logger.info(f"     python scripts/evaluate.py --model_path results/best_model.pth --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()

