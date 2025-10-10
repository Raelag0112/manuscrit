"""
Graph construction from segmented organoids
"""

import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
import networkx as nx
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Build cellular graphs from segmented organoids
    
    Cells are nodes, edges represent spatial relationships
    """
    
    def __init__(
        self,
        edge_method: str = 'knn',
        k_neighbors: int = 8,
        radius: Optional[float] = None,
        include_self_loops: bool = False,
    ):
        """
        Initialize graph builder
        
        Args:
            edge_method: 'knn', 'radius', 'delaunay', or 'combined'
            k_neighbors: Number of nearest neighbors for KNN
            radius: Radius for radius-based edges (in pixels/microns)
            include_self_loops: Add self-loops to graph
        """
        self.edge_method = edge_method
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.include_self_loops = include_self_loops
        
        logger.info(f"Initialized GraphBuilder with method: {edge_method}")
    
    def extract_cell_centroids(
        self,
        masks: np.ndarray
    ) -> np.ndarray:
        """
        Extract 3D centroids of segmented cells
        
        Args:
            masks: Labeled segmentation masks (Z, Y, X)
        
        Returns:
            centroids: Array of shape (num_cells, 3) with (z, y, x) coordinates
        """
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids != 0]  # Remove background
        
        centroids = []
        for cell_id in cell_ids:
            coords = np.where(masks == cell_id)
            centroid = np.array([
                np.mean(coords[0]),  # z
                np.mean(coords[1]),  # y
                np.mean(coords[2]),  # x
            ])
            centroids.append(centroid)
        
        return np.array(centroids)
    
    def build_knn_edges(
        self,
        positions: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build K-nearest neighbor edges
        
        Args:
            positions: Cell positions (N, 3)
            k: Number of nearest neighbors
        
        Returns:
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, D) - distances
        """
        # Build KDTree for efficient neighbor search
        tree = cKDTree(positions)
        
        # Find k+1 nearest neighbors (including self)
        distances, indices = tree.query(positions, k=k+1)
        
        # Remove self-connections
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Build edge list
        num_nodes = len(positions)
        sources = np.repeat(np.arange(num_nodes), k)
        targets = indices.flatten()
        
        edge_index = np.stack([sources, targets], axis=0)
        edge_attr = distances.flatten().reshape(-1, 1)
        
        return edge_index, edge_attr
    
    def build_radius_edges(
        self,
        positions: np.ndarray,
        radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build radius-based edges
        
        Args:
            positions: Cell positions (N, 3)
            radius: Connection radius
        
        Returns:
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, D) - distances
        """
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # Create edges for pairs within radius
        mask = (distances <= radius) & (distances > 0)
        sources, targets = np.where(mask)
        
        edge_index = np.stack([sources, targets], axis=0)
        edge_attr = distances[mask].reshape(-1, 1)
        
        return edge_index, edge_attr
    
    def build_delaunay_edges(
        self,
        positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edges based on 3D Delaunay triangulation
        
        Args:
            positions: Cell positions (N, 3)
        
        Returns:
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, D) - distances
        """
        # Compute Delaunay triangulation
        tri = Delaunay(positions)
        
        # Extract edges from simplices
        edges_set = set()
        for simplex in tri.simplices:
            # Add all edges in this tetrahedron
            for i in range(4):
                for j in range(i+1, 4):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges_set.add(edge)
        
        # Convert to arrays
        edges = np.array(list(edges_set))
        sources = edges[:, 0]
        targets = edges[:, 1]
        
        # Make edges bidirectional
        edge_index = np.concatenate([
            np.stack([sources, targets]),
            np.stack([targets, sources])
        ], axis=1)
        
        # Compute distances
        distances = np.linalg.norm(
            positions[edge_index[0]] - positions[edge_index[1]],
            axis=1
        )
        edge_attr = distances.reshape(-1, 1)
        
        return edge_index, edge_attr
    
    def build_edges(
        self,
        positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edges using configured method
        
        Args:
            positions: Cell positions (N, 3)
        
        Returns:
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, D)
        """
        if self.edge_method == 'knn':
            edge_index, edge_attr = self.build_knn_edges(positions, self.k_neighbors)
        
        elif self.edge_method == 'radius':
            if self.radius is None:
                raise ValueError("radius must be specified for radius edge method")
            edge_index, edge_attr = self.build_radius_edges(positions, self.radius)
        
        elif self.edge_method == 'delaunay':
            edge_index, edge_attr = self.build_delaunay_edges(positions)
        
        elif self.edge_method == 'combined':
            # Combine KNN and radius
            edge_index_knn, edge_attr_knn = self.build_knn_edges(
                positions, self.k_neighbors
            )
            edge_index_radius, edge_attr_radius = self.build_radius_edges(
                positions, self.radius
            )
            
            # Combine and remove duplicates
            edge_index = np.concatenate([edge_index_knn, edge_index_radius], axis=1)
            edge_attr = np.concatenate([edge_attr_knn, edge_attr_radius], axis=0)
            
            # Remove duplicate edges
            edge_set = set()
            unique_indices = []
            for i in range(edge_index.shape[1]):
                edge = (edge_index[0, i], edge_index[1, i])
                if edge not in edge_set:
                    edge_set.add(edge)
                    unique_indices.append(i)
            
            edge_index = edge_index[:, unique_indices]
            edge_attr = edge_attr[unique_indices]
        
        else:
            raise ValueError(f"Unknown edge method: {self.edge_method}")
        
        # Add self-loops if requested
        if self.include_self_loops:
            num_nodes = positions.shape[0]
            self_loops = np.arange(num_nodes).reshape(1, -1)
            self_loops = np.vstack([self_loops, self_loops])
            edge_index = np.concatenate([edge_index, self_loops], axis=1)
            
            # Add zero distances for self-loops
            zero_distances = np.zeros((num_nodes, 1))
            edge_attr = np.concatenate([edge_attr, zero_distances], axis=0)
        
        return edge_index, edge_attr
    
    def build_graph(
        self,
        masks: np.ndarray,
        node_features: Optional[np.ndarray] = None,
        label: Optional[int] = None,
    ) -> Data:
        """
        Build PyG graph from segmented organoid
        
        Args:
            masks: Labeled segmentation masks (Z, Y, X)
            node_features: Optional pre-computed node features (N, D)
            label: Optional graph-level label
        
        Returns:
            PyG Data object representing the cellular graph
        """
        # Extract cell centroids
        positions = self.extract_cell_centroids(masks)
        num_cells = len(positions)
        
        logger.info(f"Building graph with {num_cells} nodes")
        
        # Build edges
        edge_index, edge_attr = self.build_edges(positions)
        
        logger.info(f"Created {edge_index.shape[1]} edges")
        
        # Use positions as default features if none provided
        if node_features is None:
            node_features = positions
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(positions, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)
        
        # Add metadata
        data.num_cells = num_cells
        
        return data
    
    def compute_graph_statistics(
        self,
        data: Data
    ) -> Dict:
        """
        Compute graph statistics
        
        Args:
            data: PyG Data object
        
        Returns:
            Dictionary of graph statistics
        """
        edge_index = data.edge_index.numpy()
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        
        # Convert to NetworkX for analysis
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = [(edge_index[0, i], edge_index[1, i]) for i in range(num_edges)]
        G.add_edges_from(edges)
        
        # Compute statistics
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': np.mean(degree_values),
            'std_degree': np.std(degree_values),
            'min_degree': np.min(degree_values),
            'max_degree': np.max(degree_values),
            'density': nx.density(G),
        }
        
        # Add clustering coefficient if graph is not too large
        if num_nodes < 1000:
            try:
                stats['avg_clustering'] = nx.average_clustering(G)
            except:
                stats['avg_clustering'] = 0.0
        
        return stats


def build_dataset_graphs(
    segmentation_dir: str,
    output_dir: str,
    labels_file: Optional[str] = None,
    edge_method: str = 'knn',
    k_neighbors: int = 8,
) -> None:
    """
    Build graphs for entire dataset
    
    Args:
        segmentation_dir: Directory containing segmentation masks
        output_dir: Directory to save graph .pt files
        labels_file: Optional CSV with labels
        edge_method: Edge construction method
        k_neighbors: Number of neighbors for KNN
    """
    from pathlib import Path
    import pandas as pd
    
    seg_dir = Path(segmentation_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels if provided
    labels_dict = {}
    if labels_file is not None:
        df = pd.read_csv(labels_file)
        labels_dict = dict(zip(df['filename'], df['label']))
    
    # Initialize builder
    builder = GraphBuilder(edge_method=edge_method, k_neighbors=k_neighbors)
    
    # Process all masks
    mask_files = sorted(seg_dir.glob('*_masks.tif'))
    logger.info(f"Found {len(mask_files)} segmentation files")
    
    for i, mask_file in enumerate(mask_files):
        logger.info(f"Processing {i+1}/{len(mask_files)}: {mask_file.name}")
        
        # Load masks
        from cellpose import io
        masks = io.imread(str(mask_file))
        
        # Get label
        label = labels_dict.get(mask_file.stem, None)
        
        # Build graph
        data = builder.build_graph(masks, label=label)
        
        # Save
        output_path = out_dir / f"{mask_file.stem}.pt"
        torch.save(data, output_path)
        
        # Log statistics
        stats = builder.compute_graph_statistics(data)
        logger.info(f"  Stats: {stats}")
    
    logger.info("Graph construction complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build cellular graphs from segmentations')
    parser.add_argument('--segmentation_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--labels_file', type=str, default=None)
    parser.add_argument('--edge_method', type=str, default='knn',
                        choices=['knn', 'radius', 'delaunay', 'combined'])
    parser.add_argument('--k_neighbors', type=int, default=8)
    parser.add_argument('--radius', type=float, default=None)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    build_dataset_graphs(
        args.segmentation_dir,
        args.output_dir,
        args.labels_file,
        args.edge_method,
        args.k_neighbors,
    )

