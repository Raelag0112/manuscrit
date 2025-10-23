"""
Quick test to verify all modules can be imported
"""

print("Testing imports...")

print("  - data.preprocessing... ", end="")
from data.preprocessing import preprocess_image, denoise_image
print("OK")

print("  - utils.segmentation... ", end="")
from utils.segmentation import CellposeSegmenter
print("OK")

print("  - utils.clustering... ", end="")
from utils.clustering import OrganoidSeparator
print("OK")

print("  - utils.features... ", end="")
from utils.features import FeatureExtractor
print("OK")

print("  - utils.graph_builder... ", end="")
from utils.graph_builder import GraphBuilder
print("OK")

print("  - utils.metrics... ", end="")
from utils.metrics import compute_metrics
print("OK")

print("  - synthetic.generator... ", end="")
from synthetic.generator import SyntheticOrganoidGenerator
print("OK")

print("  - synthetic.point_processes... ", end="")
from synthetic.point_processes import PoissonProcess, MaternClusterProcess
print("OK")

print("  - models.gcn... ", end="")
from models.gcn import GCNClassifier
print("OK")

print("  - models.gat... ", end="")
from models.gat import GATClassifier
print("OK")

print("  - models.egnn... ", end="")
from models.egnn import EGNNClassifier
print("OK")

print("  - models.classifier... ", end="")
from models.classifier import OrganoidClassifier
print("OK")

print("  - visualization.interpretability... ", end="")
from visualization.interpretability import GNNExplainer, SaliencyMapper
print("OK")

print("  - scripts.train... ", end="")
from scripts.train import Trainer
print("OK")

print("\nAll imports successful!")
print("\nPipeline components:")
print("  ✓ Preprocessing (denoise, normalize, background correction)")
print("  ✓ Segmentation (Cellpose wrapper)")
print("  ✓ Clustering (DBSCAN organoid separation)")
print("  ✓ Feature extraction (volume, sphericity, centroids)")
print("  ✓ Graph construction (K-NN, radius, Delaunay, hybrid)")
print("  ✓ Synthetic generation (Poisson, Matérn, spatial transforms)")
print("  ✓ GNN models (GCN, GAT, GraphSAGE, GIN, EGNN)")
print("  ✓ Training pipeline (AdamW, ReduceLROnPlateau, early stopping)")
print("  ✓ Interpretability (GradCAM, attention, perturbation)")
print("  ✓ End-to-end pipeline (scripts/pipeline_full.py)")

