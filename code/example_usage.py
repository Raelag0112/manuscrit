"""
Example usage of the complete pipeline
"""

import numpy as np
from pathlib import Path

# Example 1: Preprocess an image
from data.preprocessing import preprocess_image

image = np.random.rand(100, 512, 512)  # Fake 3D image
preprocessed = preprocess_image(
    image,
    normalize=True,
    denoise=True,
    correct_bg=False,
    median_size=3,
    gaussian_sigma=0.5,
)
print(f"Preprocessed shape: {preprocessed.shape}")


# Example 2: Segment cells
from utils.segmentation import CellposeSegmenter

segmenter = CellposeSegmenter(model_type='cyto2', diameter=20)
# masks, info = segmenter.segment_3d(preprocessed)
# print(f"Segmented {info['num_cells']} cells")


# Example 3: Separate organoids with DBSCAN
from utils.clustering import OrganoidSeparator

separator = OrganoidSeparator(eps=30.0, min_samples=20)
# organoid_masks = separator.separate_organoids(masks)
# print(f"Found {len(organoid_masks)} organoids")


# Example 4: Extract features
from utils.features import FeatureExtractor

extractor = FeatureExtractor(include_morphology=True)
# features, names = extractor.extract_all_features(None, organoid_masks[0])
# print(f"Extracted {len(names)} features per cell")


# Example 5: Build graph
from utils.graph_builder import GraphBuilder

builder = GraphBuilder(edge_method='knn', k_neighbors=10)
# graph = builder.build_graph(organoid_masks[0], node_features=features)
# print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")


# Example 6: Generate synthetic data
from synthetic.generator import SyntheticOrganoidGenerator

generator = SyntheticOrganoidGenerator(radius=100.0, k_neighbors=10)
synthetic_data = generator.generate_organoid(
    num_cells=250,
    process_type='matern',
    label=1,
    feature_dim=5,
)
print(f"Synthetic organoid: {synthetic_data.num_nodes} cells")


# Example 7: Train model
from models.classifier import OrganoidClassifier

model = OrganoidClassifier.create(
    model_type='egnn',
    in_channels=5,
    num_classes=2,
    hidden_channels=256,
    num_layers=5,
    dropout=0.15,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# Example 8: Explain predictions
from visualization.interpretability import GNNExplainer

# explainer = GNNExplainer(model, device='cpu')
# explanation = explainer.explain_prediction(synthetic_data, method='gradcam')
# print(f"Top 5 important cells: {explanation['top_k_nodes'][:5]}")


# Example 9: Full pipeline
from scripts.pipeline_full import OrganoidPipeline

# pipeline = OrganoidPipeline(
#     model_path='results/best_model.pth',
#     device='cpu',
# )
# results = pipeline.process_image_file('image.tif', output_dir='output/')
# print(f"Processed {results['num_organoids']} organoids")

print("\nAll examples completed successfully!")

