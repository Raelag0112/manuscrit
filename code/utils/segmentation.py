"""
Cell segmentation using Cellpose
"""

import numpy as np
from cellpose import models, io
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class CellposeSegmenter:
    """
    Wrapper for Cellpose 3D cell segmentation
    
    Reference: Stringer et al. (2021), Nature Methods
    """
    
    def __init__(
        self,
        model_type: str = 'cyto2',
        gpu: bool = True,
        diameter: Optional[float] = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
    ):
        """
        Initialize Cellpose model
        
        Args:
            model_type: 'cyto', 'cyto2', 'nuclei', or path to custom model
            gpu: Use GPU if available
            diameter: Expected cell diameter in pixels (None = auto-estimate)
            flow_threshold: Flow error threshold (0.4 for mammalian cells)
            cellprob_threshold: Cell probability threshold (0.0 for default)
        """
        self.model_type = model_type
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        
        # Initialize model
        self.model = models.Cellpose(
            gpu=gpu,
            model_type=model_type
        )
        
        logger.info(f"Initialized Cellpose with model: {model_type}, GPU: {gpu}")
    
    def segment_3d(
        self,
        image: np.ndarray,
        channels: list = [0, 0],
        do_3D: bool = True,
        anisotropy: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Segment cells in 3D image
        
        Args:
            image: 3D array (Z, Y, X) or (Z, Y, X, C) for multi-channel
            channels: [cytoplasm_channel, nucleus_channel], [0,0] for grayscale
            do_3D: Use 3D segmentation (True) or 2D slice-by-slice (False)
            anisotropy: Ratio of Z-spacing to XY-spacing (e.g., 2.0 if Z is 2x larger)
        
        Returns:
            masks: Labeled image (Z, Y, X) with cell IDs
            flows: Dictionary with flows, probabilities, etc.
        """
        logger.info(f"Segmenting 3D image of shape {image.shape}")
        
        # Run segmentation
        masks, flows, styles, diams = self.model.eval(
            image,
            channels=channels,
            diameter=self.diameter,
            do_3D=do_3D,
            anisotropy=anisotropy,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
        )
        
        num_cells = len(np.unique(masks)) - 1  # -1 for background
        logger.info(f"Segmented {num_cells} cells")
        
        return masks, {
            'flows': flows,
            'styles': styles,
            'diameters': diams,
            'num_cells': num_cells
        }
    
    def segment_batch(
        self,
        images: list,
        channels: list = [0, 0],
        do_3D: bool = True,
    ) -> list:
        """
        Segment batch of images
        
        Args:
            images: List of 3D numpy arrays
            channels: Channel configuration
            do_3D: Use 3D segmentation
        
        Returns:
            List of (masks, info) tuples
        """
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            masks, info = self.segment_3d(image, channels, do_3D)
            results.append((masks, info))
        
        return results
    
    def segment_from_file(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_flows: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Load image, segment, and optionally save results
        
        Args:
            image_path: Path to image file (TIFF, PNG, etc.)
            output_dir: Directory to save results (None = don't save)
            save_flows: Save flow fields and probabilities
        
        Returns:
            masks: Segmentation masks
            info: Segmentation information
        """
        image_path = Path(image_path)
        logger.info(f"Loading image from {image_path}")
        
        # Load image
        image = io.imread(str(image_path))
        
        # Segment
        masks, info = self.segment_3d(image)
        
        # Save if requested
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save masks
            mask_path = output_dir / f"{image_path.stem}_masks.tif"
            io.imsave(str(mask_path), masks.astype(np.uint16))
            logger.info(f"Saved masks to {mask_path}")
            
            # Save flows
            if save_flows:
                flow_path = output_dir / f"{image_path.stem}_flows.tif"
                io.imsave(str(flow_path), info['flows'][0])
                logger.info(f"Saved flows to {flow_path}")
        
        return masks, info
    
    def compute_quality_metrics(
        self,
        masks: np.ndarray
    ) -> dict:
        """
        Compute quality metrics for segmentation
        
        Args:
            masks: Labeled segmentation masks
        
        Returns:
            Dictionary of quality metrics
        """
        num_cells = len(np.unique(masks)) - 1
        
        # Cell size statistics
        cell_sizes = []
        for cell_id in range(1, num_cells + 1):
            size = np.sum(masks == cell_id)
            cell_sizes.append(size)
        
        cell_sizes = np.array(cell_sizes)
        
        metrics = {
            'num_cells': num_cells,
            'mean_cell_size': np.mean(cell_sizes),
            'std_cell_size': np.std(cell_sizes),
            'min_cell_size': np.min(cell_sizes) if len(cell_sizes) > 0 else 0,
            'max_cell_size': np.max(cell_sizes) if len(cell_sizes) > 0 else 0,
            'median_cell_size': np.median(cell_sizes) if len(cell_sizes) > 0 else 0,
        }
        
        return metrics


def segment_organoid_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    model_type: str = 'cyto2',
    diameter: Optional[float] = None,
    file_pattern: str = '*.tif',
) -> None:
    """
    Segment all organoids in a directory
    
    Args:
        input_dir: Directory containing organoid images
        output_dir: Directory to save segmentation results
        model_type: Cellpose model type
        diameter: Expected cell diameter
        file_pattern: Glob pattern for image files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize segmenter
    segmenter = CellposeSegmenter(model_type=model_type, diameter=diameter)
    
    # Find all images
    image_files = sorted(input_dir.glob(file_pattern))
    logger.info(f"Found {len(image_files)} images to segment")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        logger.info(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
        
        try:
            masks, info = segmenter.segment_from_file(
                image_path,
                output_dir=output_dir,
                save_flows=True
            )
            
            # Compute and log metrics
            metrics = segmenter.compute_quality_metrics(masks)
            logger.info(f"  Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"  Error processing {image_path.name}: {e}")
            continue
    
    logger.info("Segmentation complete!")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment organoids with Cellpose')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing organoid images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save segmentation results')
    parser.add_argument('--model_type', type=str, default='cyto2',
                        help='Cellpose model type (cyto, cyto2, nuclei)')
    parser.add_argument('--diameter', type=float, default=None,
                        help='Expected cell diameter in pixels')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern for images')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    segment_organoid_dataset(
        args.input_dir,
        args.output_dir,
        args.model_type,
        args.diameter,
        args.pattern
    )

