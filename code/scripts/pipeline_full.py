"""
Complete end-to-end pipeline from raw images to predictions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import argparse
import logging
from tqdm import tqdm
import json

from data.preprocessing import preprocess_image
from utils.segmentation import CellposeSegmenter
from utils.clustering import OrganoidSeparator
from utils.features import FeatureExtractor
from utils.graph_builder import GraphBuilder
from models.classifier import OrganoidClassifier
from visualization.interpretability import GNNExplainer

logger = logging.getLogger(__name__)


class OrganoidPipeline:
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        segmentation_params: dict = None,
        clustering_params: dict = None,
        graph_params: dict = None,
    ):
        self.device = device
        
        self.segmenter = CellposeSegmenter(
            **(segmentation_params or {})
        )
        
        self.separator = OrganoidSeparator(
            **(clustering_params or {})
        )
        
        self.feature_extractor = FeatureExtractor(
            include_intensity=False,
            include_morphology=True,
        )
        
        self.graph_builder = GraphBuilder(
            **(graph_params or {})
        )
        
        self.model = OrganoidClassifier.load_from_checkpoint(model_path, device)
        self.model.eval()
        
        self.explainer = GNNExplainer(self.model, device)
        
        logger.info("Pipeline initialized")
    
    def process_single_image(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        explain: bool = False,
    ) -> dict:
        results = {
            'num_organoids': 0,
            'organoids': [],
        }
        
        if preprocess:
            logger.info("Step 1/6: Preprocessing...")
            image = preprocess_image(
                image,
                normalize=True,
                denoise=True,
                correct_bg=False,
            )
        
        logger.info("Step 2/6: Segmentation...")
        masks, seg_info = self.segmenter.segment_3d(image)
        logger.info(f"  Segmented {seg_info['num_cells']} cells")
        
        logger.info("Step 3/6: Clustering (DBSCAN)...")
        organoid_masks = self.separator.separate_organoids(masks)
        results['num_organoids'] = len(organoid_masks)
        logger.info(f"  Separated {len(organoid_masks)} organoids")
        
        if len(organoid_masks) == 0:
            logger.warning("No organoids found")
            return results
        
        for org_idx, organoid_mask in enumerate(organoid_masks):
            logger.info(f"Step 4/6: Processing organoid {org_idx + 1}/{len(organoid_masks)}")
            
            features, feature_names = self.feature_extractor.extract_all_features(
                None,
                organoid_mask,
            )
            
            centroid_x_idx = feature_names.index('centroid_x')
            centroid_y_idx = feature_names.index('centroid_y')
            centroid_z_idx = feature_names.index('centroid_z')
            volume_idx = feature_names.index('volume')
            sphericity_idx = feature_names.index('sphericity')
            
            node_features = np.column_stack([
                features[:, centroid_z_idx],
                features[:, centroid_y_idx],
                features[:, centroid_x_idx],
                features[:, volume_idx],
                features[:, sphericity_idx],
            ])
            
            logger.info("Step 5/6: Graph construction...")
            graph_data = self.graph_builder.build_graph(
                organoid_mask,
                node_features=node_features,
            )
            
            logger.info("Step 6/6: Classification...")
            graph_data = graph_data.to(self.device)
            
            with torch.no_grad():
                output = self.model(
                    graph_data.x,
                    graph_data.edge_index,
                    graph_data.batch,
                )
                probs = torch.softmax(output, dim=1)[0]
                predicted_class = output.argmax(dim=1).item()
                confidence = probs[predicted_class].item()
            
            organoid_result = {
                'organoid_id': org_idx,
                'num_cells': graph_data.num_nodes,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probs.cpu().numpy().tolist(),
            }
            
            if explain:
                logger.info("  Computing explanations...")
                explanation = self.explainer.explain_prediction(graph_data)
                organoid_result['explanation'] = {
                    'top_k_nodes': explanation['top_k_nodes'].tolist(),
                    'top_k_importances': explanation['top_k_importances'].tolist(),
                }
            
            results['organoids'].append(organoid_result)
            
            logger.info(f"  Prediction: class {predicted_class}, confidence {confidence:.3f}")
        
        return results
    
    def process_image_file(
        self,
        image_path: str,
        output_dir: str = None,
        explain: bool = False,
    ) -> dict:
        from cellpose import io
        
        logger.info(f"Loading image from {image_path}")
        image = io.imread(image_path)
        
        results = self.process_single_image(image, preprocess=True, explain=explain)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            result_file = output_path / f"{image_name}_results.json"
            
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved results to {result_file}")
        
        return results
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = '*.tif',
        explain: bool = False,
    ):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(input_path.glob(pattern))
        logger.info(f"Found {len(image_files)} images")
        
        all_results = []
        
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                results = self.process_image_file(
                    str(image_file),
                    output_dir=str(output_path),
                    explain=explain,
                )
                results['image_file'] = image_file.name
                all_results.append(results)
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue
        
        summary_file = output_path / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")
        
        total_organoids = sum(r['num_organoids'] for r in all_results)
        logger.info(f"Processed {len(all_results)} images, {total_organoids} organoids")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Complete organoid analysis pipeline')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--explain', action='store_true',
                        help='Generate explanations')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern for directory processing')
    
    parser.add_argument('--seg_model', type=str, default='cyto2')
    parser.add_argument('--seg_diameter', type=float, default=20.0)
    
    parser.add_argument('--cluster_eps', type=float, default=30.0)
    parser.add_argument('--cluster_min_samples', type=int, default=20)
    parser.add_argument('--min_cells', type=int, default=20)
    parser.add_argument('--max_cells', type=int, default=5000)
    
    parser.add_argument('--edge_method', type=str, default='knn')
    parser.add_argument('--k_neighbors', type=int, default=10)
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*60)
    logger.info("ORGANOID ANALYSIS PIPELINE")
    logger.info("="*60)
    
    pipeline = OrganoidPipeline(
        model_path=args.model_path,
        device=args.device,
        segmentation_params={
            'model_type': args.seg_model,
            'diameter': args.seg_diameter,
        },
        clustering_params={
            'eps': args.cluster_eps,
            'min_samples': args.cluster_min_samples,
            'min_cells': args.min_cells,
            'max_cells': args.max_cells,
        },
        graph_params={
            'edge_method': args.edge_method,
            'k_neighbors': args.k_neighbors,
        },
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        logger.info("Processing single image")
        results = pipeline.process_image_file(
            str(input_path),
            output_dir=args.output_dir,
            explain=args.explain,
        )
        
    elif input_path.is_dir():
        logger.info("Processing directory")
        results = pipeline.process_directory(
            str(input_path),
            output_dir=args.output_dir,
            pattern=args.pattern,
            explain=args.explain,
        )
    else:
        logger.error(f"Input path not found: {input_path}")
        return
    
    logger.info("="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()

