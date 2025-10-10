# Apprentissage profond pour l'analyse des organoïdes : modélisation par graphes des architectures cellulaires 3D

Code d'implémentation de la thèse d'Alexandre Martin - 2025

## 📋 Description

Ce repository contient l'implémentation complète de la pipeline d'analyse d'organoïdes par Graph Neural Networks (GNNs), incluant :

- Segmentation cellulaire 3D avec Cellpose
- Construction de graphes cellulaires spatiaux
- Architectures GNN (GCN, GAT, GraphSAGE, GIN, EGNN)
- Génération de données synthétiques via processus ponctuels spatiaux
- Entraînement, évaluation et interprétabilité

## 🏗️ Structure du projet

```
code/
├── data/                    # Gestion des données
│   ├── dataset.py          # Classes Dataset PyTorch
│   ├── loader.py           # DataLoaders
│   └── preprocessing.py    # Prétraitement d'images
├── models/                  # Architectures GNN
│   ├── gcn.py              # Graph Convolutional Network
│   ├── gat.py              # Graph Attention Network
│   ├── graphsage.py        # GraphSAGE
│   ├── gin.py              # Graph Isomorphism Network
│   ├── egnn.py             # E(n)-Equivariant GNN
│   └── classifier.py       # Modèle de classification complet
├── utils/                   # Utilitaires
│   ├── segmentation.py     # Pipeline Cellpose
│   ├── graph_builder.py    # Construction de graphes
│   ├── features.py         # Extraction de features
│   ├── metrics.py          # Métriques d'évaluation
│   └── visualization.py    # Outils de visualisation
├── synthetic/               # Génération de données synthétiques
│   ├── point_processes.py  # Processus ponctuels spatiaux
│   ├── generator.py        # Générateur d'organoïdes
│   └── statistics.py       # Statistiques spatiales (Ripley, etc.)
├── scripts/                 # Scripts d'exécution
│   ├── train.py            # Entraînement de modèles
│   ├── evaluate.py         # Évaluation
│   ├── inference.py        # Inférence sur nouvelles données
│   └── generate_data.py    # Génération de données synthétiques
├── visualization/           # Visualisation avancée
│   ├── plot_graphs.py      # Visualisation de graphes
│   ├── plot_3d.py          # Visualisation 3D d'organoïdes
│   └── interpretability.py # GNNExplainer, attention maps
├── configs/                 # Fichiers de configuration
│   ├── config.yaml         # Configuration générale
│   ├── models/             # Configs spécifiques aux modèles
│   └── experiments/        # Configs d'expériences
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_synthetic_generation.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
└── tests/                   # Tests unitaires
    ├── test_segmentation.py
    ├── test_graph_builder.py
    └── test_models.py
```

## 🚀 Installation

### Prérequis
- Python 3.9+
- CUDA 11.8+ (pour GPU, recommandé)
- 16GB+ RAM (32GB recommandé pour grands organoïdes)

### Installation des dépendances

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installation de PyTorch Geometric (si nécessaire)
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 📊 Utilisation

### 1. Génération de données synthétiques

```bash
python scripts/generate_data.py \
    --output_dir data/synthetic \
    --num_organoids 5000 \
    --processes poisson,matern,strauss \
    --num_cells_range 100,500
```

### 2. Segmentation d'organoïdes réels

```bash
python scripts/segment_organoids.py \
    --input_dir data/raw_images \
    --output_dir data/segmented \
    --model cyto2 \
    --diameter 20
```

### 3. Construction de graphes

```bash
python scripts/build_graphs.py \
    --segmentation_dir data/segmented \
    --output_dir data/graphs \
    --k_neighbors 8 \
    --radius 50
```

### 4. Entraînement

```bash
python scripts/train.py \
    --config configs/experiments/gcn_baseline.yaml \
    --data_dir data/graphs \
    --output_dir results/gcn_baseline \
    --epochs 200 \
    --batch_size 32
```

### 5. Évaluation

```bash
python scripts/evaluate.py \
    --model_path results/gcn_baseline/best_model.pth \
    --test_data data/graphs/test \
    --output_dir results/evaluation
```

### 6. Inférence sur nouvelles données

```bash
python scripts/inference.py \
    --model_path results/best_model.pth \
    --input_images data/new_organoids/*.tif \
    --output_predictions results/predictions.csv
```

## 🔬 Architectures GNN disponibles

- **GCN** : Graph Convolutional Network (Kipf & Welling, 2017)
- **GAT** : Graph Attention Network (Veličković et al., 2018)
- **GraphSAGE** : Inductive learning avec sampling (Hamilton et al., 2017)
- **GIN** : Graph Isomorphism Network (Xu et al., 2019)
- **EGNN** : E(n)-Equivariant GNN pour géométrie 3D (Satorras et al., 2021)

## 📈 Expériences

### Baseline avec données synthétiques
```bash
python scripts/train.py --config configs/experiments/synthetic_baseline.yaml
```

### Transfer learning : synthétique → réel
```bash
# 1. Pré-entraînement sur synthétique
python scripts/train.py --config configs/experiments/pretrain_synthetic.yaml

# 2. Fine-tuning sur réel
python scripts/train.py \
    --config configs/experiments/finetune_real.yaml \
    --pretrained_model results/pretrain/best_model.pth
```

### Comparaison d'architectures
```bash
bash scripts/run_architecture_comparison.sh
```

## 📊 Résultats

Les résultats d'entraînement sont sauvegardés dans :
- `results/<experiment>/` : checkpoints, logs, métriques
- TensorBoard logs : `tensorboard --logdir results/`
- Weights & Biases : tracking en ligne (optionnel)

## 🎨 Visualisation

### Visualiser un graphe cellulaire 3D
```python
from visualization.plot_3d import plot_organoid_graph

plot_organoid_graph(
    graph_path='data/graphs/organoid_001.pt',
    output_path='figures/organoid_001.html'
)
```

### Interprétabilité (GNNExplainer)
```python
from visualization.interpretability import explain_prediction

explain_prediction(
    model=trained_model,
    graph=test_graph,
    output_dir='results/explanations/'
)
```

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/

# Test spécifique
pytest tests/test_graph_builder.py -v

# Avec couverture
pytest tests/ --cov=. --cov-report=html
```

## 📚 Citation

Si vous utilisez ce code, merci de citer :

```bibtex
@phdthesis{martin2025organoids,
  title={Apprentissage profond pour l'analyse des organoïdes : modélisation par graphes des architectures cellulaires 3D},
  author={Martin, Alexandre},
  year={2025},
  school={Université Côte d'Azur}
}
```

## 📄 Licence

Ce code est fourni sous licence MIT. Voir `LICENSE` pour plus de détails.

## 👥 Auteur

**Alexandre Martin**  
Université Côte d'Azur - 2025

## 🙏 Remerciements

- Cellpose (Stringer et al., 2021) pour la segmentation
- PyTorch Geometric (Fey & Lenssen, 2019) pour les outils GNN
- Communauté PyTorch et open-source

## 📞 Contact

Pour toute question : [votre.email@example.com]

