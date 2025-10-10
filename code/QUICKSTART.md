# Guide de démarrage rapide

## Installation

```bash
cd code/
pip install -r requirements.txt
```

## Workflow complet

### 1. Générer des données synthétiques

```bash
python scripts/generate_data.py \
    --output_dir data/synthetic \
    --num_train 3000 \
    --num_val 500 \
    --num_test 500 \
    --num_cells_min 100 \
    --num_cells_max 500
```

**Résultat** : Dataset synthétique dans `data/synthetic/` avec 3 splits (train/val/test)

### 2. Entraîner un modèle

```bash
python scripts/train.py \
    --data_dir data/synthetic \
    --output_dir results/gcn_baseline \
    --model gcn \
    --hidden_channels 128 \
    --num_layers 3 \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001
```

**Résultat** : Modèle entraîné dans `results/gcn_baseline/best_model.pth`

### 3. Évaluer le modèle

```bash
python scripts/evaluate.py \
    --model_path results/gcn_baseline/best_model.pth \
    --data_dir data/synthetic \
    --output_dir results/evaluation
```

**Résultat** : Métriques et rapports dans `results/evaluation/`

### 4. Comparer différentes architectures

```bash
# GCN
python scripts/train.py --data_dir data/synthetic --output_dir results/gcn --model gcn

# GAT
python scripts/train.py --data_dir data/synthetic --output_dir results/gat --model gat

# GraphSAGE
python scripts/train.py --data_dir data/synthetic --output_dir results/graphsage --model graphsage

# GIN
python scripts/train.py --data_dir data/synthetic --output_dir results/gin --model gin

# EGNN
python scripts/train.py --data_dir data/synthetic --output_dir results/egnn --model egnn
```

## Workflow avec données réelles

### 1. Segmenter des organoïdes

```python
from utils.segmentation import CellposeSegmenter

segmenter = CellposeSegmenter(model_type='cyto2', diameter=20)
masks, info = segmenter.segment_from_file(
    'path/to/organoid.tif',
    output_dir='data/segmented/'
)
```

### 2. Construire des graphes

```python
from utils.graph_builder import GraphBuilder

builder = GraphBuilder(edge_method='knn', k_neighbors=8)
graph = builder.build_graph(masks, label=0)

import torch
torch.save(graph, 'data/graphs/organoid_001.pt')
```

### 3. Extraire des features

```python
from utils.features import FeatureExtractor

extractor = FeatureExtractor(
    include_intensity=True,
    include_morphology=True
)

features, feature_names = extractor.extract_all_features(
    image=original_image,
    masks=masks
)
```

### 4. Entraîner sur données réelles

```bash
python scripts/train.py \
    --data_dir data/real \
    --output_dir results/real \
    --model gat \
    --epochs 300
```

## Transfer Learning : Synthétique → Réel

### 1. Pré-entraînement sur synthétique

```bash
python scripts/train.py \
    --data_dir data/synthetic \
    --output_dir results/pretrain \
    --model gcn \
    --epochs 200
```

### 2. Fine-tuning sur réel

```bash
python scripts/train.py \
    --data_dir data/real \
    --output_dir results/finetune \
    --model gcn \
    --epochs 100 \
    --lr 0.0001 \
    --pretrained_model results/pretrain/best_model.pth
```

## Visualisation

### Visualiser un graphe

```python
from visualization.plot_graphs import plot_graph_3d
import torch

data = torch.load('data/synthetic/train/organoid_train_00000.pt')
plot_graph_3d(data, 'figures/organoid_graph.png')
```

### Visualisation interactive 3D

```python
from visualization.plot_3d import interactive_organoid_viewer

data_list = [torch.load(f) for f in Path('data/synthetic/train/').glob('*.pt')[:10]]
interactive_organoid_viewer(data_list)
```

### Interprétabilité

```python
from visualization.interpretability import GNNExplainer

explainer = GNNExplainer(model, device='cuda')
node_scores, predicted_class = explainer.explain(test_data)
explainer.visualize_explanation(test_data, node_scores, 'explanation.png')
```

## Notebooks Jupyter

Des notebooks d'exemple sont disponibles dans `notebooks/` :

- `01_data_exploration.ipynb` : Explorer les données
- `02_synthetic_generation.ipynb` : Générer et visualiser des données synthétiques
- `03_model_training.ipynb` : Entraîner des modèles interactivement
- `04_results_analysis.ipynb` : Analyser les résultats

## Conseils

### GPU
Pour utiliser le GPU, assurez-vous que CUDA est installé :
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Hyperparamètres recommandés

**Pour données synthétiques :**
- Model: GCN ou GAT
- Hidden channels: 128
- Num layers: 3
- Dropout: 0.5
- Learning rate: 0.001
- Batch size: 32

**Pour données réelles :**
- Model: GAT ou EGNN
- Hidden channels: 256
- Num layers: 4
- Dropout: 0.6
- Learning rate: 0.0005
- Batch size: 16

### Debugging

Si le modèle ne converge pas :
1. Vérifier que les données sont correctement normalisées
2. Réduire le learning rate (0.0001)
3. Augmenter le dropout (0.7)
4. Essayer un autre modèle (GAT souvent plus robuste)

Si OOM (out of memory) :
1. Réduire batch_size (16 ou 8)
2. Réduire hidden_channels (64)
3. Réduire num_layers (2)

## Support

Pour plus d'informations, consultez :
- README.md
- Documentation des modèles dans `models/`
- Code source avec commentaires détaillés

