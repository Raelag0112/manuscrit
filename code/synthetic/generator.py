"""
Synthetic organoid generator

Generates complete synthetic organoid datasets with:
- Cell positions via point processes
- Cell features
- Graph structures
- Labels
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .point_processes import (
    PoissonProcess,
    InhomogeneousPoissonProcess,
    MaternClusterProcess,
    StraussProcess,
    create_gradient_intensity
)
from ..utils.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class SyntheticOrganoidGenerator:
    """
    Generate synthetic organoid datasets
    """
    
    PROCESS_TYPES = {
        'poisson': PoissonProcess,
        'matern': MaternClusterProcess,
        'strauss': StraussProcess,
    }
    
    def __init__(
        self,
        radius: float = 100.0,
        edge_method: str = 'knn',
        k_neighbors: int = 8,
        seed: Optional[int] = None,
    ):
        """
        Initialize generator
        
        Args:
            radius: Organoid radius
            edge_method: Graph edge construction method
            k_neighbors: Number of neighbors for KNN
            seed: Random seed for reproducibility
        """
        self.radius = radius
        self.edge_method = edge_method
        self.k_neighbors = k_neighbors
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.graph_builder = GraphBuilder(
            edge_method=edge_method,
            k_neighbors=k_neighbors,
        )
    
    def generate_cell_positions(
        self,
        num_cells: int,
        process_type: str = 'poisson',
        **process_kwargs
    ) -> np.ndarray:
        """
        Generate cell positions using point process
        
        Args:
            num_cells: Number of cells
            process_type: Type of point process
            **process_kwargs: Additional process parameters
        
        Returns:
            positions: Array (num_cells, 3)
        """
        if process_type == 'poisson':
            process = PoissonProcess(radius=self.radius, **process_kwargs)
            positions = process.generate(num_points=num_cells)
        
        elif process_type == 'inhomogeneous':
            intensity_func = create_gradient_intensity(**process_kwargs)
            process = InhomogeneousPoissonProcess(
                intensity_func=intensity_func,
                radius=self.radius,
            )
            positions = process.generate(num_points=num_cells)
        
        elif process_type == 'matern':
            process = MaternClusterProcess(
                radius=self.radius,
                **process_kwargs
            )
            positions = process.generate(num_points=num_cells)
        
        elif process_type == 'strauss':
            process = StraussProcess(
                radius=self.radius,
                **process_kwargs
            )
            positions = process.generate(num_points=num_cells)
        
        else:
            raise ValueError(f"Unknown process type: {process_type}")
        
        return positions
    
    def generate_cell_features(
        self,
        positions: np.ndarray,
        feature_dim: int = 10,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Generate cell features based on positions
        
        Args:
            positions: Cell positions (N, 3)
            feature_dim: Number of features per cell
            add_noise: Add random noise to features
        
        Returns:
            features: Array (N, feature_dim)
        """
        num_cells = len(positions)
        features = np.zeros((num_cells, feature_dim))
        
        # Feature 0-2: Normalized positions
        features[:, :3] = positions / self.radius
        
        # Feature 3: Distance from center
        distances = np.linalg.norm(positions, axis=1)
        features[:, 3] = distances / self.radius
        
        # Feature 4: Number of nearby neighbors (local density)
        from scipy.spatial import distance_matrix
        dist_matrix = distance_matrix(positions, positions)
        radius_threshold = 20.0
        features[:, 4] = np.sum(dist_matrix < radius_threshold, axis=1) - 1
        
        # Feature 5-9: Random features
        if feature_dim > 5:
            features[:, 5:] = np.random.randn(num_cells, feature_dim - 5)
        
        # Add noise
        if add_noise:
            noise = np.random.randn(num_cells, feature_dim) * 0.1
            features = features + noise
        
        return features.astype(np.float32)
    
    def generate_organoid(
        self,
        num_cells: int,
        process_type: str,
        label: int,
        feature_dim: int = 10,
        **process_kwargs
    ) -> Data:
        """
        Generate single synthetic organoid
        
        Args:
            num_cells: Number of cells
            process_type: Point process type
            label: Class label
            feature_dim: Feature dimension
            **process_kwargs: Process parameters
        
        Returns:
            PyG Data object
        """
        # Generate positions
        positions = self.generate_cell_positions(
            num_cells,
            process_type,
            **process_kwargs
        )
        
        # Generate features
        features = self.generate_cell_features(positions, feature_dim)
        
        # Build graph
        # Create fake masks (just use cell IDs)
        fake_masks = np.arange(1, len(positions) + 1)
        
        # Build edges
        edge_index, edge_attr = self.graph_builder.build_edges(positions)
        
        # Create PyG Data
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            pos=torch.tensor(positions, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor([label], dtype=torch.long),
            num_cells=len(positions),
            process_type=process_type,
        )
        
        return data
    
    def generate_dataset(
        self,
        num_organoids: int,
        num_cells_range: Tuple[int, int] = (100, 500),
        process_distribution: Optional[Dict[str, float]] = None,
        feature_dim: int = 10,
    ) -> List[Data]:
        """
        Generate dataset of synthetic organoids
        
        Args:
            num_organoids: Number of organoids to generate
            num_cells_range: (min, max) number of cells per organoid
            process_distribution: Dict mapping process type to label
            feature_dim: Feature dimension
        
        Returns:
            List of PyG Data objects
        """
        if process_distribution is None:
            # Default: equal distribution across 3 process types
            process_distribution = {
                'poisson': 0,
                'matern': 1,
                'strauss': 2,
            }
        
        dataset = []
        processes = list(process_distribution.keys())
        labels = list(process_distribution.values())
        
        for i in range(num_organoids):
            # Sample process type
            process_type = np.random.choice(processes)
            label = process_distribution[process_type]
            
            # Sample number of cells
            num_cells = np.random.randint(num_cells_range[0], num_cells_range[1])
            
            # Set process parameters based on type
            if process_type == 'matern':
                process_kwargs = {
                    'parent_intensity': 5.0,
                    'cluster_radius': 20.0,
                    'offspring_per_parent': 15.0,
                }
            elif process_type == 'strauss':
                process_kwargs = {
                    'intensity': 100.0,
                    'interaction_radius': 15.0,
                    'beta': 0.1,
                }
            else:  # poisson
                process_kwargs = {}
            
            # Generate organoid
            data = self.generate_organoid(
                num_cells=num_cells,
                process_type=process_type,
                label=label,
                feature_dim=feature_dim,
                **process_kwargs
            )
            
            dataset.append(data)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_organoids} organoids")
        
        logger.info(f"Generated complete dataset with {len(dataset)} organoids")
        return dataset
    
    def save_dataset(
        self,
        dataset: List[Data],
        output_dir: str,
        split: str = 'train',
    ) -> None:
        """
        Save dataset to disk
        
        Args:
            dataset: List of PyG Data objects
            output_dir: Output directory
            split: Dataset split name ('train', 'val', 'test')
        """
        output_path = Path(output_dir) / split
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, data in enumerate(dataset):
            filename = output_path / f"organoid_{split}_{i:05d}.pt"
            torch.save(data, filename)
        
        logger.info(f"Saved {len(dataset)} organoids to {output_path}")
    
    def generate_and_save(
        self,
        output_dir: str,
        num_train: int = 3000,
        num_val: int = 500,
        num_test: int = 500,
        **generation_kwargs
    ) -> None:
        """
        Generate and save train/val/test splits
        
        Args:
            output_dir: Output directory
            num_train: Number of training samples
            num_val: Number of validation samples
            num_test: Number of test samples
            **generation_kwargs: Passed to generate_dataset
        """
        logger.info("Generating training set...")
        train_data = self.generate_dataset(num_train, **generation_kwargs)
        self.save_dataset(train_data, output_dir, 'train')
        
        logger.info("Generating validation set...")
        val_data = self.generate_dataset(num_val, **generation_kwargs)
        self.save_dataset(val_data, output_dir, 'val')
        
        logger.info("Generating test set...")
        test_data = self.generate_dataset(num_test, **generation_kwargs)
        self.save_dataset(test_data, output_dir, 'test')
        
        logger.info(f"Dataset generation complete! Saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic organoid dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--num_train', type=int, default=3000)
    parser.add_argument('--num_val', type=int, default=500)
    parser.add_argument('--num_test', type=int, default=500)
    parser.add_argument('--num_cells_min', type=int, default=100)
    parser.add_argument('--num_cells_max', type=int, default=500)
    parser.add_argument('--feature_dim', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    generator = SyntheticOrganoidGenerator(seed=args.seed)
    
    generator.generate_and_save(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        num_cells_range=(args.num_cells_min, args.num_cells_max),
        feature_dim=args.feature_dim,
    )

