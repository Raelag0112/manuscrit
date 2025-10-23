"""
DBSCAN clustering for separating individual organoids
"""

import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OrganoidSeparator:
    
    def __init__(
        self,
        eps: float = 30.0,
        min_samples: int = 20,
        min_cells: int = 20,
        max_cells: int = 5000,
        border_margin: float = 0.05,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.border_margin = border_margin
    
    def extract_centroids(self, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids != 0]
        
        centroids = []
        cell_id_list = []
        
        for cell_id in cell_ids:
            coords = np.where(masks == cell_id)
            centroid = np.array([
                np.mean(coords[0]),
                np.mean(coords[1]),
                np.mean(coords[2]),
            ])
            centroids.append(centroid)
            cell_id_list.append(cell_id)
        
        return np.array(centroids), np.array(cell_id_list)
    
    def is_near_border(self, centroid: np.ndarray, image_shape: Tuple) -> bool:
        z, y, x = centroid
        zmax, ymax, xmax = image_shape
        
        margin_z = self.border_margin * zmax
        margin_y = self.border_margin * ymax
        margin_x = self.border_margin * xmax
        
        if z < margin_z or z > zmax - margin_z:
            return True
        if y < margin_y or y > ymax - margin_y:
            return True
        if x < margin_x or x > xmax - margin_x:
            return True
        
        return False
    
    def cluster_cells(
        self,
        centroids: np.ndarray,
    ) -> np.ndarray:
        if len(centroids) == 0:
            return np.array([])
        
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(centroids)
        
        return labels
    
    def filter_clusters(
        self,
        labels: np.ndarray,
        centroids: np.ndarray,
        image_shape: Tuple,
    ) -> Dict[int, np.ndarray]:
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        valid_clusters = {}
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size < self.min_cells or cluster_size > self.max_cells:
                continue
            
            cluster_centroids = centroids[cluster_mask]
            mean_centroid = np.mean(cluster_centroids, axis=0)
            
            if self.is_near_border(mean_centroid, image_shape):
                continue
            
            valid_clusters[label] = cluster_mask
        
        return valid_clusters
    
    def separate_organoids(
        self,
        masks: np.ndarray,
    ) -> List[np.ndarray]:
        centroids, cell_ids = self.extract_centroids(masks)
        
        if len(centroids) == 0:
            logger.warning("No cells found in masks")
            return []
        
        labels = self.cluster_cells(centroids)
        
        valid_clusters = self.filter_clusters(labels, centroids, masks.shape)
        
        if len(valid_clusters) == 0:
            logger.warning("No valid organoids found after filtering")
            return []
        
        organoid_masks = []
        
        for cluster_label, cluster_mask in valid_clusters.items():
            cluster_cell_ids = cell_ids[cluster_mask]
            
            organoid_mask = np.zeros_like(masks)
            for new_id, cell_id in enumerate(cluster_cell_ids, start=1):
                organoid_mask[masks == cell_id] = new_id
            
            organoid_masks.append(organoid_mask)
        
        logger.info(f"Separated {len(organoid_masks)} organoids from {len(centroids)} cells")
        
        return organoid_masks
    
    def get_statistics(
        self,
        organoid_masks: List[np.ndarray],
    ) -> Dict:
        num_organoids = len(organoid_masks)
        
        if num_organoids == 0:
            return {
                'num_organoids': 0,
                'mean_cells_per_organoid': 0,
                'std_cells_per_organoid': 0,
                'min_cells': 0,
                'max_cells': 0,
            }
        
        cells_per_organoid = []
        for mask in organoid_masks:
            num_cells = len(np.unique(mask)) - 1
            cells_per_organoid.append(num_cells)
        
        cells_per_organoid = np.array(cells_per_organoid)
        
        stats = {
            'num_organoids': num_organoids,
            'mean_cells_per_organoid': float(np.mean(cells_per_organoid)),
            'std_cells_per_organoid': float(np.std(cells_per_organoid)),
            'min_cells': int(np.min(cells_per_organoid)),
            'max_cells': int(np.max(cells_per_organoid)),
        }
        
        return stats


def separate_organoids_from_file(
    mask_file: str,
    output_dir: str,
    eps: float = 30.0,
    min_samples: int = 20,
    min_cells: int = 20,
    max_cells: int = 5000,
) -> None:
    from pathlib import Path
    from cellpose import io
    
    mask_path = Path(mask_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    masks = io.imread(str(mask_path))
    
    separator = OrganoidSeparator(
        eps=eps,
        min_samples=min_samples,
        min_cells=min_cells,
        max_cells=max_cells,
    )
    
    organoid_masks = separator.separate_organoids(masks)
    
    stats = separator.get_statistics(organoid_masks)
    logger.info(f"Statistics: {stats}")
    
    for i, organoid_mask in enumerate(organoid_masks):
        output_file = output_path / f"{mask_path.stem}_organoid_{i+1}.tif"
        io.imsave(str(output_file), organoid_mask.astype(np.uint16))
        logger.info(f"Saved organoid {i+1} to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--eps', type=float, default=30.0)
    parser.add_argument('--min_samples', type=int, default=20)
    parser.add_argument('--min_cells', type=int, default=20)
    parser.add_argument('--max_cells', type=int, default=5000)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    separate_organoids_from_file(
        args.mask_file,
        args.output_dir,
        args.eps,
        args.min_samples,
        args.min_cells,
        args.max_cells,
    )

