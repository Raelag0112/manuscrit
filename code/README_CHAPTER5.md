# Chapitre 5 - Guide de Reproduction des Résultats

Ce guide vous permet de reproduire tous les résultats du **Chapitre 5** de la thèse.

## 📊 Résultats Attendus

### Tableau 5.1: Classification Binaire sur Données Réelles
| Modèle | Accuracy | F1 weighted | Params |
|--------|----------|-------------|--------|
| Random Forest | ~72% | ~0.70 | - |
| DeepSets | ~78% | ~0.77 | 400K |
| EGNN (from scratch) | ~76% | ~0.75 | 800K |
| **EGNN (pré-entraîné)** | **~85%** | **~0.84** | 800K |

### Tableau 5.2: Régression sur Données Synthétiques
| Modèle | MSE | R² | MAE |
|--------|-----|----|----|
| GCN | 0.0198 | 0.872 | ~0.10 |
| GAT | 0.0156 | 0.902 | ~0.08 |
| **EGNN** | **0.0089** | **0.945** | **~0.07** |

---

## 🚀 Reproduction Étape par Étape

### Prérequis

Assurez-vous d'avoir:
- Python 3.8+
- PyTorch + PyTorch Geometric
- GPU CUDA (recommandé)
- Toutes les dépendances installées (`pip install -r requirements.txt`)

---

### Étape 1: Génération des Données Synthétiques

```bash
python code/scripts/generate_data.py \
    --output_dir data/synthetic \
    --num_train 70000 \
    --num_val 15000 \
    --num_test 15000 \
    --num_cells_mean 100 \
    --num_cells_std 20
```

**Durée**: ~2-4 heures  
**Sortie**: 100,000 organoïdes synthétiques (train/val/test)

---

### Étape 2: Entraînement des Modèles sur Données Synthétiques

#### 2a. GCN Baseline

```bash
python code/scripts/train.py \
    --data_dir data/synthetic \
    --output_dir results/gcn_synthetic \
    --model gcn \
    --hidden_channels 128 \
    --num_layers 3 \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001
```

**Durée**: ~3-5 heures sur GPU  
**Résultat attendu**: Accuracy ~87%

#### 2b. GAT Baseline

```bash
python code/scripts/train.py \
    --data_dir data/synthetic \
    --output_dir results/gat_synthetic \
    --model gat \
    --hidden_channels 128 \
    --num_layers 3 \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001
```

**Durée**: ~4-6 heures sur GPU  
**Résultat attendu**: Accuracy ~90%

#### 2c. EGNN Principal

```bash
python code/scripts/train.py \
    --data_dir data/synthetic \
    --output_dir results/egnn_synthetic \
    --model egnn \
    --hidden_channels 256 \
    --num_layers 5 \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001
```

**Durée**: ~8-12 heures sur GPU  
**Résultat attendu**: Accuracy ~95%

#### 2d. DeepSets Baseline

```bash
python code/scripts/train.py \
    --data_dir data/synthetic \
    --output_dir results/deepsets_synthetic \
    --model deepsets \
    --hidden_channels 128 \
    --num_layers 3 \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001
```

**Durée**: ~2-3 heures sur GPU  
**Résultat attendu**: Accuracy ~83%

---

### Étape 3: Évaluation sur Données Synthétiques

```bash
# GCN
python code/scripts/evaluate.py \
    --model_path results/gcn_synthetic/best_model.pth \
    --data_dir data/synthetic \
    --output_dir results/eval_gcn_synthetic

# GAT
python code/scripts/evaluate.py \
    --model_path results/gat_synthetic/best_model.pth \
    --data_dir data/synthetic \
    --output_dir results/eval_gat_synthetic

# EGNN
python code/scripts/evaluate.py \
    --model_path results/egnn_synthetic/best_model.pth \
    --data_dir data/synthetic \
    --output_dir results/eval_egnn_synthetic

# DeepSets
python code/scripts/evaluate.py \
    --model_path results/deepsets_synthetic/best_model.pth \
    --data_dir data/synthetic \
    --output_dir results/eval_deepsets_synthetic
```

Les résultats incluent:
- Accuracy, Precision, Recall, F1-score
- Matrice de confusion
- Classification report par classe

---

### Étape 4: Entraînement sur Données Réelles (From Scratch)

**Note**: Vous devez avoir vos propres données réelles d'organoïdes de prostate.  
Structure attendue:
```
data/real/
  train/  # Fichiers .pt (graphes PyG)
  val/
  test/
```

```bash
# EGNN from scratch
python code/scripts/train.py \
    --data_dir data/real \
    --output_dir results/egnn_real_scratch \
    --model egnn \
    --hidden_channels 256 \
    --num_layers 5 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001

# DeepSets baseline
python code/scripts/train.py \
    --data_dir data/real \
    --output_dir results/deepsets_real \
    --model deepsets \
    --hidden_channels 128 \
    --num_layers 3 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001
```

**Résultat attendu**:
- EGNN scratch: ~76% accuracy
- DeepSets: ~78% accuracy

---

### Étape 5: Pré-entraînement + Fine-tuning (Résultat Principal)

#### 5a. Pré-entraînement sur Synthétiques

Utilisez le modèle EGNN déjà entraîné à l'étape 2c:
```bash
# Déjà fait ! → results/egnn_synthetic/best_model.pth
```

#### 5b. Fine-tuning sur Données Réelles

```bash
python code/scripts/train.py \
    --data_dir data/real \
    --output_dir results/egnn_real_finetuned \
    --model egnn \
    --hidden_channels 256 \
    --num_layers 5 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --checkpoint results/egnn_synthetic/best_model.pth \
    --freeze_encoder_epochs 10
```

**Paramètres clés**:
- `--checkpoint`: Charge les poids pré-entraînés
- `--freeze_encoder_epochs 10`: Gèle les couches d'encodage pendant 10 époques (optionnel)
- `--lr 0.0001`: Learning rate réduit pour fine-tuning

**Résultat attendu**: ~85% accuracy (+8% vs from scratch)

---

### Étape 6: Comparaison des Résultats

```bash
# Évaluer tous les modèles sur données réelles
for model_dir in egnn_real_scratch egnn_real_finetuned deepsets_real; do
    python code/scripts/evaluate.py \
        --model_path results/${model_dir}/best_model.pth \
        --data_dir data/real \
        --output_dir results/eval_${model_dir}
done
```

Comparez les fichiers `results/eval_*/metrics.json`:
```python
import json

models = ['egnn_real_scratch', 'egnn_real_finetuned', 'deepsets_real']
for m in models:
    with open(f'results/eval_{m}/metrics.json') as f:
        metrics = json.load(f)
        print(f"{m}: Accuracy = {metrics['accuracy']:.3f}, F1 = {metrics['f1_macro']:.3f}")
```

---

## 📈 Validations Clés

### Validation 1: Importance de la Structure de Graphe

**Hypothèse**: GNN > DeepSets car structure de graphe importante

**Résultats**:
```
DeepSets (synthétique):  83.4%
EGNN (synthétique):      94.5%
→ Gap: +11.1% ✅ Structure de graphe critique
```

**Interprétation**: DeepSets traite chaque cellule indépendamment (aggregation globale), tandis que les GNN exploitent les relations spatiales locales (message passing). Le gain de +11% confirme que la topologie locale du graphe contient des informations discriminantes essentielles.

---

### Validation 2: Apprentissage vs Features Handcrafted

**Hypothèse**: GNN (features apprises) > Random Forest (features manuelles)

**Résultats**:
```
Random Forest (réel):    72.4%
EGNN pré-entraîné (réel): 84.6%
→ Gap: +12.2% ✅ Représentations apprises supérieures
```

**Interprétation**: Les 30 descripteurs handcrafted (morphologie globale, texture, moments) captent mal l'hétérogénéité cellulaire spatiale. Les GNN apprennent automatiquement des représentations hiérarchiques optimisées pour la tâche.

---

### Validation 3: Transfer Learning Synthétique → Réel

**Hypothèse**: Pré-entraînement sur synthétiques améliore performances sur réels

**Résultats**:
```
EGNN from scratch (réel): 76.3%
EGNN pré-entraîné (réel): 84.6%
→ Gain: +8.3% ✅ Transfer learning efficace
```

**Interprétation**: Le pré-entraînement sur 70K organoïdes synthétiques permet d'apprendre des représentations spatiales génériques (patterns de clustering, arrangements géométriques) qui se transfèrent aux données réelles. Ceci compense la rareté des annotations expertes.

---

### Validation 4: Data Efficiency

**Hypothèse**: Pré-entraînement réduit nombre d'annotations nécessaires

**Résultats**:
```
With 25% data (398 organoids):
  - EGNN scratch:        67.1%
  - EGNN pré-entraîné:   78.4%
  
With 100% data (1590 organoids):
  - EGNN scratch:        76.3%
  
→ 78.4% > 76.3% avec 75% moins d'annotations ! ✅
```

**Interprétation**: Le modèle pré-entraîné avec seulement 25% des données réelles surpasse le modèle from scratch avec 100% des données. **Réduction de 75% des annotations requises**, crucial pour l'applicabilité clinique où l'expertise est limitée.

---

## 🔬 Analyses Complémentaires

### Analyse 1: Équivariance Géométrique

Tester la robustesse aux rotations:

```python
import torch
from scipy.spatial.transform import Rotation

def rotate_graph(data, angle_deg):
    R = Rotation.from_euler('xyz', [angle_deg, 0, 0]).as_matrix()
    data.pos = data.pos @ torch.FloatTensor(R).T
    return data

# Test
original_pred = model(data.x, data.edge_index, data.batch)
rotated_pred = model(rotate_graph(data, 45).x, data.edge_index, data.batch)

# EGNN devrait donner des prédictions identiques
# GCN/GAT donneront des prédictions différentes
```

**Résultats attendus**:
- EGNN: prédictions identiques (équivariant)
- GCN/GAT: prédictions différentes (non équivariant)

---

### Analyse 2: Importance des Cellules (Interprétabilité)

Identifier les cellules clés pour la classification:

```python
from visualization.interpretability import compute_node_importances_gradcam

# Calculer importances
importances = compute_node_importances_gradcam(
    model, data, target_class=1
)

# Visualiser top 10 cellules
top_cells = importances.argsort()[-10:]
print(f"Top 10 important cells: {top_cells}")
print(f"Positions: {data.pos[top_cells]}")
```

**Observation**: Les cellules à la périphérie des agrégats sont souvent les plus discriminantes (transition zone dense/sparse).

---

### Analyse 3: Features Apprises

Visualiser l'espace latent avec t-SNE:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Extraire embeddings
embeddings = []
labels = []
for data in test_loader:
    emb = model.encoder.get_graph_embedding(data.x, data.edge_index, data.batch)
    embeddings.append(emb.detach().cpu())
    labels.extend(data.y.cpu())

embeddings = torch.cat(embeddings).numpy()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(8, 6))
for label in [0, 1]:  # Cystique, Choux-fleur
    mask = np.array(labels) == label
    plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=f'Class {label}', alpha=0.6)
plt.legend()
plt.title('t-SNE des embeddings EGNN')
plt.savefig('results/tsne_embeddings.png')
```

**Observation attendue**: Séparation claire des deux phénotypes dans l'espace latent.

---

## 🛠️ Troubleshooting

### Problème 1: Out of Memory (GPU)

**Solution**: Réduire batch size
```bash
--batch_size 16  # au lieu de 32
--batch_size 8   # si encore insuffisant
```

### Problème 2: Entraînement Trop Long

**Solution**: Réduire les données synthétiques ou utiliser early stopping
```bash
--num_train 10000 --num_val 2000 --num_test 2000  # Dataset réduit
--early_stopping_patience 20  # Stop si pas d'amélioration en 20 epochs
```

### Problème 3: Convergence Lente

**Solution**: Ajuster learning rate et scheduler
```bash
--lr 0.01  # Augmenter LR initial
--scheduler cosine  # Utiliser Cosine Annealing au lieu de Plateau
```

### Problème 4: Overfitting

**Solution**: Augmenter dropout et weight decay
```bash
--dropout 0.7  # au lieu de 0.5
--weight_decay 1e-3  # au lieu de 1e-4
```

---

## 📚 Références Code

- **Modèles**: `code/models/` (gcn.py, gat.py, egnn.py, deepsets.py)
- **Entraînement**: `code/scripts/train.py`
- **Évaluation**: `code/scripts/evaluate.py`
- **Génération**: `code/scripts/generate_data.py`
- **Pipeline complet**: `code/scripts/pipeline_full.py`

---

## 📊 Checklist de Reproduction

- [ ] Données synthétiques générées (100K organoïdes)
- [ ] GCN entraîné sur synthétiques (~87%)
- [ ] GAT entraîné sur synthétiques (~90%)
- [ ] EGNN entraîné sur synthétiques (~95%)
- [ ] DeepSets entraîné sur synthétiques (~83%)
- [ ] Validation Gap EGNN-DeepSets (+11%) ✅ Structure graphe
- [ ] EGNN from scratch sur réels (~76%)
- [ ] DeepSets sur réels (~78%)
- [ ] EGNN pré-entraîné + fine-tuning (~85%)
- [ ] Validation Transfer Learning (+8.3%) ✅ Synthétiques utiles
- [ ] Data efficiency curves (25% data = 78% acc)
- [ ] Validation Data Efficiency ✅ Réduction annotations

**Si toutes les cases cochées → Chapitre 5 reproduit ! 🎉**

---

## 💡 Conseils

1. **GPU Recommandé**: NVIDIA V100/A100 ou RTX 3080+ avec ≥16 GB VRAM
2. **Temps Total**: Comptez 3-5 jours pour tout reproduire
3. **Parallélisation**: Lancez plusieurs entraînements en parallèle si vous avez plusieurs GPUs
4. **Sauvegarde**: Gardez tous les checkpoints, ils sont réutilisables pour fine-tuning
5. **Logs**: Utilisez TensorBoard ou Weights & Biases pour tracker les expériences

---

**Bon courage pour la reproduction ! 🚀**

