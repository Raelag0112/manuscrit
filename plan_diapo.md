# Plan du Diaporama de Soutenance

**Titre** : Deep Learning for Organoid Analysis: Graph-Based Modeling of 3D Cellular Architectures  
**Candidat** : Alexandre Martin  
**Date** : 17 décembre 2025  
**Durée** : 45 minutes

---

## PARTIE 1 : INTRODUCTION (6-7 min) — Slides 1-10

### Slide 1 : Titre (30 sec)
**Contenu textuel :**
- Titre de la thèse
- Nom du candidat
- Date de soutenance
- Logos : INRIA, ANR Morpheus, IPMC, Paris Cité

**Visuel :** Background élégant avec image 3D d'organoïde (img/3Dviz.png en transparence)

---

### Slide 2 : Accroche — Les organoïdes révolutionnent la biologie (1 min)
**Contenu textuel :**
- "Mini-organs grown in a dish"
- Self-organized 3D structures from stem cells
- Bridge between 2D cultures and animal models
- Applications: personalized medicine, drug screening, disease modeling

**Visuel :** Montage photos de différents types d'organoïdes (intestin, cerveau, prostate)

---

### Slide 3 : Nos organoïdes de prostate (1 min)
**Contenu textuel :**
- Two main phenotypes:
  - **Cystic** (healthy): spherical, central cavity, organized structure
  - **Cauliflower** (perturbed): irregular surface, buds, disorganized
- Dataset: 2,272 organoids collected over 22 months
- Collaboration: ANR Morpheus, IPMC Nice, Paris Cité

**Visuel :** img/3Dviz.png — Visualisation 3D comparative cystique vs choux-fleurs

---

### Slide 4 : Le verrou scientifique #1 — Quantification automatique (45 sec)
**Contenu textuel :**
- Manual analysis: 15-30 min per organoid → NOT SCALABLE
- Subjective, variable between observers
- Cannot handle high-throughput screening (thousands of organoids)

**Visuel :** Icônes représentant temps, variabilité, non-reproductibilité

---

### Slide 5 : Le verrou scientifique #2 — Données annotées rares (45 sec)
**Contenu textuel :**
- No public organoid datasets (unlike ImageNet)
- Expert annotation is expensive and time-consuming
- Limited reproducibility across laboratories

**Visuel :** Comparaison visuelle : ImageNet (14M images) vs notre dataset (2K organoïdes)

---

### Slide 6 : Le verrou scientifique #3 — Robustesse géométrique (45 sec)
**Contenu textuel :**
- Organoids have no preferred orientation in culture
- Predictions must be invariant to rotations/translations
- Acquisition variations between laboratories

**Visuel :** Animation conceptuelle : même organoïde sous différentes orientations → même prédiction

---

### Slide 7 : Questions de recherche (1 min)
**Contenu textuel :**
- **Q1:** Are geometric graphs an effective representation?
- **Q2:** Do equivariant GNNs outperform classical approaches?
- **Q3:** Is transfer learning from synthetic data effective?

**Visuel :** 3 icônes illustrant chaque question

---

### Slide 8 : Plan de la présentation (30 sec)
**Contenu textuel :**
1. **Theoretical Foundations**: Graphs & GNNs
2. **Methodology**: End-to-end pipeline
3. **Experimental Results**: Synthetic & real data
4. **Conclusion & Perspectives**

**Visuel :** Frise chronologique visuelle des 4 parties

---

## PARTIE 2 : FONDEMENTS THÉORIQUES (12-14 min) — Slides 9-22

### Slide 9 : Titre Partie 2 — Graphs and Graph Neural Networks (10 sec)
**Contenu textuel :**
- "From Images to Graphs: A Natural Abstraction"

**Visuel :** Transition avec image organoïde → graphe

---

### Slide 10 : Limites des CNN 3D (1.5 min)
**Contenu textuel :**
- **Memory footprint**: ~2 GB per organoid → prohibitive
- **Downsampling**: destroys fine cellular details
- **No native geometric invariance**: requires extensive data augmentation
- **Loss of relational structure**: treats organoid as voxel grid

**Visuel :** Schéma comparatif : Image 3D (2GB) → Downsampled (perte info) vs Graph (10 MB, structure préservée)

---

### Slide 11 : Représentation par graphes — Une abstraction naturelle (1.5 min)
**Contenu textuel :**
- Cells form a **network of spatial interactions**
- Relational organization determines phenotype
- Abstraction: Cell → Node, Neighborhood → Edge
- **Spectacular compression**: GB → MB (factor 1000×)

**Visuel :** Schéma en 3 étapes : Image 3D → Point cloud → Graph (img/graph_comparison.png)

---

### Slide 12 : Introduction aux Graph Neural Networks (2 min)
**Contenu textuel :**
- **Problem**: How to learn on non-Euclidean data?
- **Message Passing paradigm**:
  - Each node aggregates information from its neighbors
  - Formula: h_i^{l+1} = UPDATE(h_i^l, AGGREGATE({h_j^l : j ∈ N(i)}))
  - Stacking L layers → receptive field of radius L

**Visuel :** Animation schématique du message passing sur 3 itérations

---

### Slide 13 : Pooling global et classification (1 min)
**Contenu textuel :**
- **Global pooling**: Aggregate node features to graph-level representation
  - Mean pooling: h_G = (1/N) Σ h_i
  - Max pooling: h_G = max(h_i)
- **Classification head**: MLP(h_G) → class prediction

**Visuel :** Schéma : Graph → Node embeddings → Pooling → MLP → Prediction

---

### Slide 14 : Architecture GCN — Baseline (1 min)
**Contenu textuel :**
- **Graph Convolutional Network** (Kipf & Welling, 2017)
- Normalized mean aggregation of neighbors
- Simple but treats all neighbors equally
- h_i^{l+1} = σ(Σ_j (1/√d_i d_j) W h_j^l)

**Visuel :** Schéma GCN avec formule

---

### Slide 15 : Architecture GAT — Our main choice (1.5 min)
**Contenu textuel :**
- **Graph Attention Network** (Veličković et al., 2018)
- **Attention mechanism**: learns to weight neighbors
- Multi-head attention: captures different relationship types
- α_ij = softmax(LeakyReLU(a^T[Wh_i || Wh_j]))

**Visuel :** Schéma GAT avec visualisation des poids d'attention sur un graphe

---

### Slide 16 : Architecture DeepSets — Comparison baseline (1 min)
**Contenu textuel :**
- **Set-based approach** (Zaheer et al., 2017)
- Global aggregation without explicit neighborhood structure
- Baseline to evaluate the contribution of local structure
- Surprisingly good performance → global statistics are informative

**Visuel :** Schéma DeepSets : φ → AGG → ρ

---

### Slide 17 : Équivariance géométrique — Concept clé (1.5 min)
**Contenu textuel :**
- **Why crucial for organoids?**
  - No preferred orientation in 3D culture
  - Prediction must not depend on arbitrary orientation
- **Invariance** vs **Equivariance**:
  - Invariance: f(T(x)) = f(x) → identical output
  - Equivariance: f(T(x)) = T(f(x)) → output transforms coherently

**Visuel :** Schéma illustratif : Rotation appliquée → même prédiction (invariance) ou transformation cohérente (équivariance)

---

### Slide 18 : Le groupe E(3) (1 min)
**Contenu textuel :**
- **E(3)** = Rotations + Translations + Reflections
- Natural symmetry group for 3D biological structures
- Guarantee by construction, not learned via augmentation

**Visuel :** Illustrations des 3 types de transformations

---

### Slide 19 : Architecture EGNN — Équivariance garantie (1.5 min)
**Contenu textuel :**
- **Equivariant Graph Neural Network** (Satorras et al., 2021)
- **Invariant messages**: Uses only distances (invariant quantities)
- **Equivariant coordinate update**: Positions transform coherently
- Advantages:
  - No data augmentation needed for rotations
  - Better generalization with less data
  - Robustness guaranteed by construction

**Visuel :** Schéma architecture EGNN avec flux d'information

---

### Slide 20 : EGNN — Formules clés (1 min)
**Contenu textuel :**
- **Messages**: m_ij = φ_e([h_i || h_j || ||x_i - x_j||²])
- **Coordinate update**: x_i' = x_i + Σ_j (x_i - x_j) · φ_x(m_ij)
- **Feature update**: h_i' = φ_h([h_i || Σ_j m_ij])

**Visuel :** Formules avec code couleur pour chaque composant

---

### Slide 21 : Comparaison des architectures (1 min)
**Contenu textuel :**
| Architecture | Key Feature | Trade-off |
|--------------|-------------|-----------|
| GCN | Simple baseline | Equal weighting |
| GAT | Attention | Best performance |
| DeepSets | Global aggregation | No local structure |
| EGNN | E(3)-equivariance | Geometric robustness |

**Visuel :** Tableau comparatif avec icônes

---

### Slide 22 : Résumé Partie 2 (30 sec)
**Contenu textuel :**
- Graphs capture relational structure (1000× compression)
- GNNs learn from non-Euclidean data
- GAT: best performance via attention
- EGNN: geometric robustness via equivariance

**Visuel :** Synthèse visuelle des 4 points clés

---

## PARTIE 3 : MÉTHODOLOGIE (8-10 min) — Slides 23-32

### Slide 23 : Titre Partie 3 — End-to-End Pipeline (10 sec)
**Contenu textuel :**
- "From Raw Images to Phenotype Predictions"

**Visuel :** Icône pipeline

---

### Slide 24 : Vue d'ensemble du pipeline (1.5 min)
**Contenu textuel :**
1. 3D confocal image (~2 GB) → Preprocessing
2. Cell segmentation (Faster Cellpose)
3. DBSCAN clustering → K-NN graph construction
4. GNN classification → Phenotype prediction
- **Total time**: ~20 min/organoid (dominated by segmentation)
- **Compression**: 1000× (GB → MB)

**Visuel :** Schéma flux complet avec volumes de données et temps à chaque étape

---

### Slide 25 : Dataset collaboratif (1 min)
**Contenu textuel :**
- **ANR Morpheus collaboration**: IPMC Nice + Paris Cité
- 1,311 samples imaged → 2,272 organoids extracted
- **Selected for study**: 500 well-differentiated organoids (~250 per class)
- Phenotypes: Cystic vs Cauliflower

**Visuel :** img/cumulative_organoids.png — Timeline de collecte

---

### Slide 26 : Optimisation de la segmentation (1.5 min)
**Contenu textuel :**
- **Problem**: Cellpose standard = 2,500 hours for 1,000 organoids!
- **Our contribution: Faster Cellpose**
  - Knowledge distillation (Teacher → Student)
  - 30% weight pruning
  - Optimized inference (batch size, mixed precision)
- **Result**: 5× faster, F1=0.95 preserved

**Visuel :** img/Comp.png — Comparaison qualitative des segmentations

---

### Slide 27 : Comparaison des méthodes de segmentation (1 min)
**Contenu textuel :**
| Method | F1 | Time/slice | Total 1000 org |
|--------|-----|-----------|----------------|
| Geometric (ellipses) | 0.88 | 3s | 250h |
| **Faster Cellpose** | **0.95** | **6s** | **500h** |
| Cellpose original | 0.98 | 30s | 2500h |

**Visuel :** Tableau avec mise en évidence de notre choix

---

### Slide 28 : Construction des graphes géométriques (1.5 min)
**Contenu textuel :**
- **Node features**: 3D coordinates (x, y, z) + volume → 4D
- **Edge construction**: K-NN (k=10) + radial cutoff
- **Symmetrization**: (i,j) edge ⇔ (j,i) edge

**Visuel :** img/graph_comparison.png — Visualisation de graphes construits

---

### Slide 29 : Génération synthétique par processus ponctuels (1.5 min)
**Contenu textuel :**
- **Motivation**: Address limited annotated data
- **Point processes on sphere**:
  - Homogeneous Poisson → Cystic phenotype (random distribution)
  - Matérn cluster → Cauliflower phenotype (aggregated cells)
- **100,000 synthetic organoids generated**

**Visuel :** img/distrib2.png — Comparaison Poisson vs Matérn sur sphère

---

### Slide 30 : Du synthétique au réel — Transfer Learning (1 min)
**Contenu textuel :**
- **Strategy**:
  1. Pre-training on 70,000 synthetic organoids (regression of parent number)
  2. Fine-tuning on 500 real organoids (binary classification)
- **Benefits**:
  - Reduced annotation needs (4× data efficiency)
  - 3× faster convergence

**Visuel :** img/Modelisation.png — Synthétiques vs réels

---

### Slide 31 : Architecture finale (1 min)
**Contenu textuel :**
- **Encoder**: GAT/EGNN (5 layers, 256 hidden dim)
- **Global pooling**: Mean + Max concatenated
- **Classification head**: MLP (256 → 128 → 2)
- **Training**: AdamW, LR=10⁻³, dropout=0.15

**Visuel :** Schéma architecture avec dimensions

---

### Slide 32 : Résumé Partie 3 (30 sec)
**Contenu textuel :**
- Complete automated pipeline: image → prediction
- Faster Cellpose: 5× speedup, F1=0.95
- Synthetic generation via point processes
- Transfer learning from synthetic to real

**Visuel :** Résumé visuel en 4 points

---

## PARTIE 4 : RÉSULTATS EXPÉRIMENTAUX (10-12 min) — Slides 33-46

### Slide 33 : Titre Partie 4 — Experimental Results (10 sec)
**Contenu textuel :**
- "Validating the Approach"

**Visuel :** Icône résultats

---

### Slide 34 : Étude comparative GRETSI 2025 — Protocole (1 min)
**Contenu textuel :**
- **Goal**: GNN vs Classical spatial statistics
- **Protocol**: Synthetic data with perfect ground truth
- **Noise types**: Gaussian noise + salt-and-pepper noise
- **Metrics**: Accuracy under varying noise levels

**Visuel :** Schéma protocole expérimental

---

### Slide 35 : Résultat 1 — Robustesse au bruit (1.5 min)
**Contenu textuel :**
- **Observation**: Spatial statistics (Ripley's K) MORE ROBUST to Gaussian noise
- GNN: more sensitive but recovers with more data
- **Optimal GNN depth**: 5-6 layers (over-smoothing beyond)

**Visuel :** img/noise.png — Courbes accuracy vs bruit gaussien

---

### Slide 36 : Résultat 2 — Généralisation géométrique (1.5 min)
**Contenu textuel :**
- **Test**: Train on spheres, test on ellipsoids (aspect ratio 2:1 to 5:1)
- **Spatial statistics**: 1.0 → 0.65 (35% drop!)
- **GNN**: 0.95 → 0.82 (graceful degradation)
- **Conclusion**: GNNs adapted to variable real morphologies

**Visuel :** img/accuracy_vs_ratio.png — Courbes accuracy vs ratio d'aspect

---

### Slide 37 : Leçon de l'étude comparative (1 min)
**Contenu textuel :**
- **GNNs**: Better geometric flexibility (variable morphologies)
- **Spatial statistics**: Better noise robustness (spherical geometries)
- **For real organoids**: Variable shapes → GNN preferred

**Visuel :** Tableau récapitulatif des forces/faiblesses

---

### Slide 38 : Performances sur données synthétiques (1.5 min)
**Contenu textuel :**
- **Task**: Regression of Matérn parent number (degree of aggregation)
- **Results on 15,000 test organoids**:

| Architecture | MSE | Gain vs GCN |
|--------------|-----|-------------|
| GCN | 0.198 | — |
| DeepSets | 0.145 | +27% |
| EGNN | 0.137 | +31% |
| **GAT** | **0.118** | **+40%** |

**Visuel :** Tableau avec mise en évidence GAT

---

### Slide 39 : Analyse des résultats synthétiques (1 min)
**Contenu textuel :**
- **GAT wins**: Attention mechanism = adaptive weighting
- **DeepSets surprisingly good**: Global statistics are informative
- **EGNN**: Trade-off performance/geometric robustness
- **Ablation**: Equivariance divides MSE by 2.8×

**Visuel :** Graphique barres comparatif MSE

---

### Slide 40 : Performances sur données réelles (1.5 min)
**Contenu textuel :**
- **Dataset**: 500 well-differentiated organoids (~250 per class)
- **Main result: 84% accuracy** (GAT pre-trained)
- **Per-class performance**:
  - Cauliflower: Precision 93%, Recall 74%
  - Cystic: Precision 78%, Recall 95%

**Visuel :** Tableau précision/rappel par classe

---

### Slide 41 : Matrice de confusion (1 min)
**Contenu textuel :**
- **Test set**: 75 organoids
- Correctly classified: 63/75 = 84%
- Main confusion: Cauliflower → Cystic (10 cases)
  - Low-deformation cauliflower organoids

**Visuel :** Matrice de confusion 2×2 avec heatmap

---

### Slide 42 : Impact du Transfer Learning (1.5 min)
**Contenu textuel :**
| Strategy | Accuracy | Gain |
|----------|----------|------|
| GAT from scratch | 76% | — |
| **GAT pre-trained** | **84%** | **+8%** |

- **Data efficiency**: 4× (125 org pre-trained ≈ 500 org from scratch)
- **Convergence**: 3× faster

**Visuel :** img/learning_curves.png — Courbes d'apprentissage comparatives

---

### Slide 43 : Courbes d'apprentissage détaillées (1 min)
**Contenu textuel :**
| % Data | From scratch | Pre-trained | Gain |
|--------|--------------|-------------|------|
| 10% (50 org) | 58% | 71% | +13% |
| 25% (125 org) | 67% | 78% | +11% |
| 50% (250 org) | 72% | 82% | +10% |
| 100% (500 org) | 76% | 84% | +8% |

**Visuel :** img/learning_curves_detailed.png — Graphique accuracy vs % données

---

### Slide 44 : Efficacité computationnelle (1 min)
**Contenu textuel :**
- **Inference throughput**: 200+ organoids/minute (GPU batch)
- **Memory footprint**: ~8 GB GPU
- **Reproducibility**: Perfect (deterministic predictions)
- **Scalability**: 1000 organoids in 17h (20 GPUs)

**Visuel :** Tableau comparatif efficacité

---

### Slide 45 : Synthèse des résultats (1 min)
**Contenu textuel :**
1. GNNs offer better geometric flexibility than spatial statistics
2. GAT: best performance (MSE=0.118 synthetic, 84% real)
3. Transfer learning: 4× data efficiency, 3× faster convergence
4. Pipeline: 200+ organoids/min, fully automated

**Visuel :** 4 icônes résumant les conclusions

---

### Slide 46 : Limitations identifiées (1 min)
**Contenu textuel :**
- Dependence on segmentation quality
- Generalization to other organoid types: to be validated
- No inter-laboratory validation
- Limited interpretability analysis

**Visuel :** Icônes représentant chaque limitation

---

## PARTIE 5 : CONCLUSION ET PERSPECTIVES (6-8 min) — Slides 47-55

### Slide 47 : Titre Partie 5 — Conclusion & Perspectives (10 sec)
**Contenu textuel :**
- "Contributions and Future Directions"

**Visuel :** Icône conclusion

---

### Slide 48 : Contribution 1 — Pipeline automatisé (1 min)
**Contenu textuel :**
- **End-to-end automated pipeline**
- Raw image → Prediction
- 1000× compression preserving biological information
- Open-source code available

**Visuel :** Schéma pipeline simplifié

---

### Slide 49 : Contribution 2 — Optimisation segmentation (1 min)
**Contenu textuel :**
- **Faster Cellpose**: 5× faster
- **Geometric method**: 15× faster (F1=0.88)
- Enables practical high-throughput screening

**Visuel :** Comparaison temps/qualité

---

### Slide 50 : Contribution 3 — Graphes géométriques & GNNs (1 min)
**Contenu textuel :**
- **Explicit relational structure capture**
- Comparison of 4 architectures: GAT, DeepSets, EGNN, GCN
- E(3)-equivariant architectures for geometric robustness

**Visuel :** Schéma graphe avec embeddings

---

### Slide 51 : Contribution 4 — Génération synthétique (1 min)
**Contenu textuel :**
- **Point processes for controlled generation**
- 100,000 synthetic organoids with known labels
- Effective transfer learning: +8% accuracy

**Visuel :** img/synthetic.png — Exemples synthétiques

---

### Slide 52 : Contribution 5 — Étude comparative GRETSI (45 sec)
**Contenu textuel :**
- **Rigorous comparison**: GNN vs spatial statistics
- GNN: better geometric generalization
- Statistics: better noise robustness
- Published at GRETSI 2025

**Visuel :** Logo GRETSI + résultat clé

---

### Slide 53 : Perspectives à court terme (1.5 min)
**Contenu textuel :**
- **Methodological extensions**:
  - Multi-scale graphs (cell → region → organoid)
  - Graph Transformers architecture
  - Morphological characterization via alpha-shapes
- **Multi-modal integration**:
  - Spatial transcriptomics + imaging
  - Temporal data (time-lapse)

**Visuel :** Schéma des extensions prévues

---

### Slide 54 : Perspectives à long terme (1 min)
**Contenu textuel :**
- **Therapeutic response prediction**
  - Patient-derived organoids → treatment testing
  - Personalized precision medicine
- **Generative graph models**
  - Rational organoid design *in silico*
  - Morphological space exploration
- **Societal impact**
  - Reduction of animal experimentation (3Rs principle)
  - Accelerated drug development

**Visuel :** Vision schématique médecine personnalisée

---

### Slide 55 : Message final (1 min)
**Contenu textuel :**
- **Geometric GNNs**: Powerful tool for 3D organoid analysis
- **Biology-AI synergy**: Virtuous circle of improvement
- **Open-source code**: Available for the community
- **Thank you!**

**Visuel :** Image 3D organoïde + remerciements

---

## SLIDES DE BACKUP (pour les questions)

### Backup 1 : Détails mathématiques EGNN
- Formules complètes avec dérivations
- Preuve d'équivariance

### Backup 2 : Over-smoothing et solutions
- Problème d'over-smoothing dans GNN profonds
- Residual connections, normalization

### Backup 3 : Comparaison avec Graph Transformers
- Attention globale vs locale
- Complexité computationnelle

### Backup 4 : Complexité computationnelle
- O(N) vs O(N²) pour différentes architectures
- Scaling aux grands graphes

### Backup 5 : Processus ponctuels — Détails
- Fonctions K, F, G de Ripley
- Tests statistiques (KS)

### Backup 6 : Détails Faster Cellpose
- Architecture distillée
- Paramètres de pruning

### Backup 7 : Dataset complet
- Distribution des 2272 organoïdes
- Critères de sélection des 500

### Backup 8 : Généralisation autres organoïdes
- Stratégie d'adaptation
- Organoïdes cérébraux, hépatiques

---

## LISTE DES IMAGES À UTILISER

### Images disponibles (dossier img/)
1. `3Dviz.png` — Visualisation 3D organoïdes cystique vs choux-fleurs
2. `3Dreco.png` — Reconstruction 3D par ellipses
3. `accuracy_vs_ratio.png` — Généralisation géométrique
4. `alphashape.png` — Alpha-shapes
5. `Comp.png` — Comparaison méthodes segmentation
6. `cumulative_organoids.png` — Timeline collecte dataset
7. `distrib2.png` — Distributions Poisson vs Matérn
8. `graph_comparison.png` — Comparaison graphes
9. `learning_curves.png` — Courbes d'apprentissage
10. `learning_curves_detailed.png` — Courbes détaillées
11. `Modelisation.png` — Modélisation synthétique
12. `noise.png` — Robustesse au bruit gaussien
13. `pepper.png` — Robustesse au bruit poivre-sel
14. `synthetic.png` — Organoïdes synthétiques

### Images à créer
1. Schéma message passing animé
2. Architecture GAT avec attention
3. Schéma pipeline complet
4. Matrice de confusion 2×2
5. Tableau comparatif architectures
6. Schéma équivariance/invariance
7. Timeline perspectives

---

## TIMING RECOMMANDÉ

| Section | Slides | Durée | Cumul |
|---------|--------|-------|-------|
| Introduction | 1-8 | 7 min | 7 min |
| Fondements GNN | 9-22 | 13 min | 20 min |
| Méthodologie | 23-32 | 9 min | 29 min |
| Résultats | 33-46 | 11 min | 40 min |
| Conclusion | 47-55 | 7 min | 47 min |
| **Total** | **55 slides** | **~47 min** | |

> **Note**: Prévoir 2-3 minutes de marge pour les transitions et ajustements.

