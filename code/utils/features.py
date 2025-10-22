"""
Feature extraction from segmented cells
"""

import numpy as np
from scipy import ndimage
from skimage import measure
import torch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract morphological and intensity features from segmented cells
    """
    
    def __init__(
        self,
        include_intensity: bool = True,
        include_morphology: bool = True,
        include_texture: bool = False,
    ):
        """
        Initialize feature extractor
        
        Args:
            include_intensity: Extract intensity statistics
            include_morphology: Extract morphological features
            include_texture: Extract texture features (expensive)
        """
        self.include_intensity = include_intensity
        self.include_morphology = include_morphology
        self.include_texture = include_texture
    
    def extract_morphological_features(
        self,
        masks: np.ndarray,
        cell_id: int
    ) -> Dict[str, float]:
        """
        Extract morphological features for a single cell
        
        Args:
            masks: Labeled segmentation (Z, Y, X)
            cell_id: ID of the cell to analyze
        
        Returns:
            Dictionary of morphological features
        """
        cell_mask = (masks == cell_id).astype(int)
        
        # Basic properties
        volume = np.sum(cell_mask)
        
        # Centroid
        coords = np.where(cell_mask)
        centroid = np.array([
            np.mean(coords[0]),
            np.mean(coords[1]),
            np.mean(coords[2]),
        ])
        
        # Bounding box
        bbox_min = np.array([np.min(coords[0]), np.min(coords[1]), np.min(coords[2])])
        bbox_max = np.array([np.max(coords[0]), np.max(coords[1]), np.max(coords[2])])
        bbox_size = bbox_max - bbox_min + 1
        
        # Surface area (approximate)
        # Use erosion to find boundary voxels
        eroded = ndimage.binary_erosion(cell_mask)
        boundary = cell_mask - eroded
        surface_area = np.sum(boundary)
        
        # Sphericity: sphere-like score
        # Perfect sphere has sphericity = 1
        if surface_area > 0:
            r_equivalent = (3 * volume / (4 * np.pi)) ** (1/3)
            sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area
        else:
            sphericity = 0.0
        
        # Compactness
        if surface_area > 0:
            compactness = volume / (surface_area ** (3/2))
        else:
            compactness = 0.0
        
        # Elongation (ratio of bbox dimensions)
        elongation = np.max(bbox_size) / (np.min(bbox_size) + 1e-8)
        
        features = {
            'volume': float(volume),
            'surface_area': float(surface_area),
            'sphericity': float(sphericity),
            'compactness': float(compactness),
            'elongation': float(elongation),
            'bbox_x': float(bbox_size[2]),
            'bbox_y': float(bbox_size[1]),
            'bbox_z': float(bbox_size[0]),
            'centroid_x': float(centroid[2]),
            'centroid_y': float(centroid[1]),
            'centroid_z': float(centroid[0]),
        }
        
        return features
    
    def extract_intensity_features(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        cell_id: int
    ) -> Dict[str, float]:
        """
        Extract intensity statistics for a single cell
        
        Args:
            image: Original intensity image (Z, Y, X)
            masks: Labeled segmentation (Z, Y, X)
            cell_id: ID of the cell to analyze
        
        Returns:
            Dictionary of intensity features
        """
        cell_mask = (masks == cell_id)
        intensities = image[cell_mask]
        
        if len(intensities) == 0:
            return {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'min_intensity': 0.0,
                'max_intensity': 0.0,
                'median_intensity': 0.0,
                'q25_intensity': 0.0,
                'q75_intensity': 0.0,
            }
        
        features = {
            'mean_intensity': float(np.mean(intensities)),
            'std_intensity': float(np.std(intensities)),
            'min_intensity': float(np.min(intensities)),
            'max_intensity': float(np.max(intensities)),
            'median_intensity': float(np.median(intensities)),
            'q25_intensity': float(np.percentile(intensities, 25)),
            'q75_intensity': float(np.percentile(intensities, 75)),
        }
        
        return features
    
    def extract_texture_features(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        cell_id: int
    ) -> Dict[str, float]:
        """
        Extract texture features (Haralick features, etc.)
        
        Args:
            image: Original intensity image (Z, Y, X)
            masks: Labeled segmentation (Z, Y, X)
            cell_id: ID of the cell to analyze
        
        Returns:
            Dictionary of texture features
        """
        # Simple texture features: gradient magnitudes
        cell_mask = (masks == cell_id)
        
        # Compute gradients
        gz, gy, gx = np.gradient(image.astype(float))
        gradient_mag = np.sqrt(gz**2 + gy**2 + gx**2)
        
        gradients = gradient_mag[cell_mask]
        
        if len(gradients) == 0:
            return {
                'mean_gradient': 0.0,
                'std_gradient': 0.0,
            }
        
        features = {
            'mean_gradient': float(np.mean(gradients)),
            'std_gradient': float(np.std(gradients)),
        }
        
        return features
    
    def extract_cell_features(
        self,
        image: Optional[np.ndarray],
        masks: np.ndarray,
        cell_id: int
    ) -> Dict[str, float]:
        """
        Extract all features for a single cell
        
        Args:
            image: Original intensity image (Z, Y, X), None if only morphology
            masks: Labeled segmentation (Z, Y, X)
            cell_id: ID of the cell to analyze
        
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Morphological features
        if self.include_morphology:
            morph_features = self.extract_morphological_features(masks, cell_id)
            features.update(morph_features)
        
        # Intensity features
        if self.include_intensity and image is not None:
            intensity_features = self.extract_intensity_features(image, masks, cell_id)
            features.update(intensity_features)
        
        # Texture features
        if self.include_texture and image is not None:
            texture_features = self.extract_texture_features(image, masks, cell_id)
            features.update(texture_features)
        
        return features
    
    def extract_all_features(
        self,
        image: Optional[np.ndarray],
        masks: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for all cells
        
        Args:
            image: Original intensity image (Z, Y, X)
            masks: Labeled segmentation (Z, Y, X)
        
        Returns:
            features: Array of shape (num_cells, num_features)
            feature_names: List of feature names
        """
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids != 0]  # Remove background
        
        num_cells = len(cell_ids)
        logger.info(f"Extracting features for {num_cells} cells")
        
        # Extract features for first cell to get feature names
        first_features = self.extract_cell_features(image, masks, cell_ids[0])
        feature_names = list(first_features.keys())
        num_features = len(feature_names)
        
        # Initialize feature matrix
        features = np.zeros((num_cells, num_features))
        
        # Extract features for all cells
        for i, cell_id in enumerate(cell_ids):
            cell_features = self.extract_cell_features(image, masks, cell_id)
            features[i] = [cell_features[name] for name in feature_names]
        
        logger.info(f"Extracted {num_features} features per cell")
        
        return features, feature_names
    
    def normalize_features(
        self,
        features: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features to zero mean and unit variance
        
        Args:
            features: Feature matrix (N, D)
            mean: Pre-computed mean (for test set), None to compute
            std: Pre-computed std (for test set), None to compute
        
        Returns:
            normalized_features: Normalized features
            mean: Feature means
            std: Feature stds
        """
        if mean is None:
            mean = np.mean(features, axis=0)
        if std is None:
            std = np.std(features, axis=0)
            std[std == 0] = 1.0  # Avoid division by zero
        
        normalized = (features - mean) / std
        
        return normalized, mean, std


def extract_dataset_features(
    image_dir: str,
    segmentation_dir: str,
    output_dir: str,
    include_intensity: bool = True,
    normalize: bool = True,
) -> None:
    """
    Extract features for entire dataset
    
    Args:
        image_dir: Directory containing original images
        segmentation_dir: Directory containing segmentation masks
        output_dir: Directory to save features
        include_intensity: Include intensity features
        normalize: Normalize features
    """
    from pathlib import Path
    from cellpose import io
    
    image_dir = Path(image_dir)
    seg_dir = Path(segmentation_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = FeatureExtractor(
        include_intensity=include_intensity,
        include_morphology=True,
        include_texture=False
    )
    
    # Process all masks
    mask_files = sorted(seg_dir.glob('*_masks.tif'))
    logger.info(f"Found {len(mask_files)} segmentation files")
    
    all_features = []
    feature_names = None
    
    for i, mask_file in enumerate(mask_files):
        logger.info(f"Processing {i+1}/{len(mask_files)}: {mask_file.name}")
        
        # Load masks
        masks = io.imread(str(mask_file))
        
        # Load image if needed
        if include_intensity:
            image_file = image_dir / mask_file.name.replace('_masks', '')
            if image_file.exists():
                image = io.imread(str(image_file))
            else:
                logger.warning(f"Image not found: {image_file}")
                image = None
        else:
            image = None
        
        # Extract features
        features, names = extractor.extract_all_features(image, masks)
        all_features.append(features)
        
        if feature_names is None:
            feature_names = names
        
        # Save individual features
        output_path = out_dir / f"{mask_file.stem}_features.npy"
        np.save(output_path, features)
    
    # Normalize if requested
    if normalize and len(all_features) > 0:
        all_features_concat = np.concatenate(all_features, axis=0)
        _, mean, std = extractor.normalize_features(all_features_concat)
        
        # Save normalization statistics
        np.save(out_dir / 'feature_mean.npy', mean)
        np.save(out_dir / 'feature_std.npy', std)
        
        logger.info("Saved normalization statistics")
    
    # Save feature names
    with open(out_dir / 'feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract cell features')
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--segmentation_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--no_intensity', action='store_true',
                        help='Exclude intensity features')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip normalization')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    extract_dataset_features(
        args.image_dir,
        args.segmentation_dir,
        args.output_dir,
        include_intensity=not args.no_intensity,
        normalize=not args.no_normalize,
    )

