# Organoid Analysis Pipeline

Complete implementation of the methodology described in **Chapter 4** and experimental baselines from **Chapter 5**.

## Pipeline Coverage

✅ **Chapter 4: 100% Complete** - All pipeline components implemented  
✅ **Chapter 5: DeepSets Integrated** - Baseline for graph structure validation

> **📊 To reproduce Chapter 5 results**, see [README_CHAPTER5.md](README_CHAPTER5.md)

### Components

1. **Preprocessing** (`data/preprocessing.py`)
   - Normalization (P1-P99 percentiles)
   - 3D median filter (3×3×3)
   - 3D gaussian filter (σ=0.5)
   - Background correction

2. **Segmentation** (`utils/segmentation.py`)
   - Cellpose wrapper (standard)
   - 3D segmentation support
   - Batch processing

3. **Organoid Separation** (`utils/clustering.py`) **[NEW]**
   - DBSCAN clustering (eps=30, min_samples=20)
   - Size filtering (20-5000 cells)
   - Border rejection
   - Multiple organoids per image

4. **Feature Extraction** (`utils/features.py`)
   - Centroids 3D (x, y, z)
   - Volume (voxel count)
   - Sphericity (π^(1/3)(6V)^(2/3)/S)
   - Vector: [x, y, z, volume, sphericity]^T ∈ ℝ^5

5. **Graph Construction** (`utils/graph_builder.py`)
   - K-NN (k=10)
   - Radius-based
   - Delaunay triangulation
   - Hybrid strategy

6. **Synthetic Generation** (`synthetic/generator.py`)
   - Poisson homogeneous (CSR)
   - Matérn cluster process
   - Spatial transformation (p'_i = p_i * (1 + α|N_r(i)|/|N_r^ref|))
   - Dataset: 70k/15k/15k split

7. **Models** (`models/`)
   - **EGNN** (E(n)-equivariant, 5 layers, 256 dim) - Main model
   - **GCN, GAT, GraphSAGE, GIN** - GNN baselines
   - **DeepSets** - Permutation-invariant baseline (Chapter 5)

8. **Training** (`scripts/train.py`)
   - AdamW optimizer
   - ReduceLROnPlateau scheduler
   - Early stopping
   - Cross-entropy loss

9. **Evaluation** (`scripts/evaluate.py`)
   - Accuracy, F1-score
   - Confusion matrix
   - Per-class metrics

10. **Interpretability** (`visualization/interpretability.py`) **[NEW]**
    - GradCAM for GNN
    - Attention visualization
    - Perturbation analysis
    - Saliency maps 3D

## Quick Start

### Generate Synthetic Data
```bash
python scripts/generate_data.py \
    --output_dir data/synthetic \
    --num_train 70000 \
    --num_val 15000 \
    --num_test 15000
```

### Train Model

**EGNN (main model)**:
```bash
python scripts/train.py \
    --data_dir data/synthetic \
    --output_dir results/egnn \
    --model egnn \
    --hidden_channels 256 \
    --num_layers 5 \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001
```

**DeepSets (baseline for comparison)**:
```bash
python scripts/train.py \
    --data_dir data/synthetic \
    --output_dir results/deepsets \
    --model deepsets \
    --hidden_channels 128 \
    --num_layers 3 \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001
```

### Full Pipeline (Raw Image → Prediction)
```bash
python scripts/pipeline_full.py \
    --model_path results/best_model.pth \
    --input image.tif \
    --output_dir output/ \
    --explain
```

### Batch Processing
```bash
python scripts/pipeline_full.py \
    --model_path results/best_model.pth \
    --input images_dir/ \
    --output_dir output/ \
    --pattern "*.tif"
```

## Module Usage

### Separate Organoids
```python
from utils.clustering import OrganoidSeparator

separator = OrganoidSeparator(eps=30.0, min_samples=20)
organoid_masks = separator.separate_organoids(masks)
```

### Generate Explanations
```python
from visualization.interpretability import GNNExplainer

explainer = GNNExplainer(model, device='cuda')
explanation = explainer.explain_prediction(data, method='gradcam')
important_cells = explanation['top_k_nodes']
```

### Preprocess Image
```python
from data.preprocessing import preprocess_image

clean_image = preprocess_image(
    raw_image,
    normalize=True,
    denoise=True,
    median_size=3,
    gaussian_sigma=0.5,
)
```

## Configuration

Edit `configs/config.yaml`:
- Model type (gcn, gat, egnn)
- Hidden dimensions (256)
- Number of layers (5)
- K-neighbors (10)
- Batch size (32)
- Learning rate (0.001)

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- Cellpose 2.2+
- scikit-learn, scipy, numpy

## References

- Chapter 4: Methodology and Processing Pipeline
- EGNN: Satorras et al. (2021)
- Cellpose: Stringer et al. (2021)
- Point Processes: Matérn (1960), Ripley (1977)

