# 📊 Figures à créer pour la thèse

Ce document liste toutes les figures prioritaires à créer, avec leurs spécifications détaillées.

---

## 🔴 **PRIORITÉ 1 : Figures essentielles (URGENT)**

### **Figure 4.1 : Pipeline complet de bout en bout** ⭐⭐⭐
**Emplacement :** Chapitre 4, Section 4.1 (Vue d'ensemble du pipeline)  
**Type :** Flowchart horizontal multi-étapes  

**Contenu :**
```
[Image 3D TIFF] 
    ↓ (2048×2048×200, 2 Go)
[Prétraitement]
    • Normalisation intensités
    • Débruitage (Gaussian σ=1)
    ↓
[Segmentation - Faster Cellpose]
    • Détection noyaux
    • F1=0.95, 0.5h/échantillon
    ↓
[Extraction point cloud]
    • Centroides cellules
    • Features morphologiques (27D)
    ↓
[Clustering DBSCAN]
    • Séparation organoïdes
    • eps=30μm, min_samples=20
    ↓
[Construction graphe]
    • K-NN (k=10)
    • ~250 nœuds, ~2500 arêtes
    ↓
[EGNN Classification]
    • 3 layers, 128 hidden dim
    • Output: 4 classes
    ↓
[Prédiction phénotype]
    • Chouxfleurs / Cystiques / Compact / Kératinisés
```

**Annotations :**
- Tailles données à chaque étape
- Temps de calcul (0.5h segmentation, 0.1s classification)
- Compression : 2 Go → 5 Mo graphe

**Fichier LaTeX :**
```latex
\begin{figure}[htbp]
\centering
% TODO: Insérer image flowchart_pipeline.pdf
\includegraphics[width=\textwidth]{chapitre4/img/flowchart_pipeline.pdf}
\caption{Pipeline complet d'analyse d'organoïdes 3D. De l'image confocale brute (2 Go) à la prédiction de phénotype, chaque étape compresse et enrichit l'information structurelle. La représentation graphe finale (5 Mo) préserve les relations spatiales biologiquement pertinentes tout en réduisant drastiquement la dimensionnalité.}
\label{fig:pipeline_complet}
\end{figure}
```

---

### **Figure 5.3 : Matrice de confusion 4×4 (dataset réel)** ⭐⭐⭐
**Emplacement :** Chapitre 5, Section 5.4.1 (ligne 567)  
**Type :** Heatmap matrice confusion  

**Données :**
```
              Prédit →
Vrai ↓     Choux  Cyst  Comp  Kérat
Chouxfleurs  194    13    3     1      (211 total)
Cystiques     11   107    4     1      (123)
Compact        2     1    3     0      (6)
Kératinisés    0     1    0     1      (2)
```

**Spécifications visuelles :**
- Colormap : Vert (correct, diagonale) → Blanc → Rouge (erreurs)
- Annotations : Nombres dans cellules + pourcentages
- Diagonale en gras
- Lignes/colonnes totaux

**Code Python pour génération :**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

confusion_matrix = np.array([
    [194, 13, 3, 1],
    [11, 107, 4, 1],
    [2, 1, 3, 0],
    [0, 1, 0, 1]
])

labels = ['Chouxfleurs', 'Cystiques', 'Compact', 'Kératinisés']

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=labels, yticklabels=labels, 
            cbar_kws={'label': 'Nombre de prédictions'},
            linewidths=0.5, linecolor='gray', ax=ax)
ax.set_xlabel('Prédit', fontsize=12, fontweight='bold')
ax.set_ylabel('Vrai', fontsize=12, fontweight='bold')
ax.set_title('Matrice de confusion - EGNN pré-entraîné\nTest set (342 organoïdes)', 
             fontsize=14, fontweight='bold')

# Accuracy totale
acc = np.trace(confusion_matrix) / confusion_matrix.sum()
ax.text(0.5, -0.15, f'Accuracy globale: {acc:.1%}', 
        transform=ax.transAxes, ha='center', fontsize=11, style='italic')

plt.tight_layout()
plt.savefig('confusion_matrix.pdf', dpi=300, bbox_inches='tight')
```

**Fichier LaTeX :**
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.75\textwidth]{chapitre5/img/confusion_matrix.pdf}
\caption{Matrice de confusion du modèle EGNN pré-entraîné sur le test set (342 organoïdes). La diagonale forte (305/342 = 89.2\%) démontre une bonne capacité discriminative. Les confusions principales (24 cas) surviennent entre Chouxfleurs et Cystiques, reflétant une ambiguïté biologique réelle rapportée par les experts.}
\label{fig:confusion_matrix_reel}
\end{figure}
```

---

### **Figure 5.4 : Barplot comparaison 5 méthodes** ⭐⭐⭐
**Emplacement :** Chapitre 5, Section 5.4.3 (après ligne 655)  
**Type :** Barplot groupé avec erreurs

**Données :**
```
Méthode                | Accuracy  | F1 weighted  | Temps (h)
--------------------- |-----------|--------------|----------
Random Forest         | 72.4±3.1% | 0.701±0.028  | 0.008
CNN 3D                | 81.2±2.8% | 0.806±0.024  | 12.0
EGNN from scratch     | 76.3±3.4% | 0.754±0.031  | 1.5
GAT                   | 79.1±2.6% | 0.782±0.025  | 1.8
EGNN pré-entraîné     | 84.6±2.1% | 0.843±0.019  | 0.5 (ft)
```

**Spécifications visuelles :**
- 3 groupes de barres (Accuracy, F1, Log Temps)
- Barres d'erreur (écart-type)
- EGNN pré-entraîné en couleur différente (vert)
- Ligne pointillée "Expertise humaine" à 81.3% (Expert 1)

**Code Python :**
```python
import matplotlib.pyplot as plt
import numpy as np

methods = ['Random\nForest', 'CNN 3D', 'EGNN\nscratch', 'GAT', 'EGNN\npre-trained']
accuracy = [72.4, 81.2, 76.3, 79.1, 84.6]
accuracy_std = [3.1, 2.8, 3.4, 2.6, 2.1]
f1 = [0.701, 0.806, 0.754, 0.782, 0.843]
f1_std = [0.028, 0.024, 0.031, 0.025, 0.019]

colors = ['#95a5a6', '#3498db', '#e74c3c', '#9b59b6', '#27ae60']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
x = np.arange(len(methods))
bars1 = ax1.bar(x, accuracy, yerr=accuracy_std, capsize=5, color=colors, 
                edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.axhline(81.3, color='red', linestyle='--', linewidth=2, label='Expert 1 (81.3%)')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=10)
ax1.set_ylim(65, 90)
ax1.grid(axis='y', alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_title('Performances de classification', fontsize=13, fontweight='bold')

# F1 weighted
bars2 = ax2.bar(x, f1, yerr=f1_std, capsize=5, color=colors, 
                edgecolor='black', linewidth=1.5, alpha=0.8)
ax2.set_ylabel('F1 weighted', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=10)
ax2.set_ylim(0.65, 0.90)
ax2.grid(axis='y', alpha=0.3)
ax2.set_title('Score F1 pondéré', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_methods.pdf', dpi=300, bbox_inches='tight')
```

**Fichier LaTeX :**
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{chapitre5/img/comparison_methods.pdf}
\caption{Comparaison des performances de 5 approches de classification sur OrganoProstate-2K. EGNN pré-entraîné surpasse toutes les méthodes (84.6\% accuracy, F1=0.843), y compris l'expertise humaine individuelle (Expert 1 : 81.3\%). Le pré-entraînement sur données synthétiques améliore significativement les performances (+8.3 points vs from scratch) tout en réduisant le temps de fine-tuning (0.5h vs 1.5h).}
\label{fig:comparison_methods}
\end{figure}
```

---

### **Figure 5.5 : Courbes data efficiency (learning curves)** ⭐⭐⭐
**Emplacement :** Chapitre 5, Section 5.4.4 (ligne 663)  
**Type :** Line plot avec 2 courbes

**Données :**
```
% données  |  From scratch  |  Pre-trained  |  Gain
10%        |  58.3%         |  71.2%        |  +12.9%
25%        |  67.1%         |  78.4%        |  +11.3%
50%        |  72.8%         |  82.1%        |  +9.3%
75%        |  75.2%         |  83.7%        |  +8.5%
100%       |  76.3%         |  84.6%        |  +8.3%
```

**Spécifications visuelles :**
- Axe X : % données (log scale possible)
- Axe Y : Accuracy (%)
- 2 courbes : From scratch (rouge), Pre-trained (vert)
- Marqueurs : points ronds
- Zone grisée : gains du pré-entraînement
- Annotations : gain à 25% (+11.3 points)

**Code Python :**
```python
import matplotlib.pyplot as plt
import numpy as np

pct_data = [10, 25, 50, 75, 100]
scratch = [58.3, 67.1, 72.8, 75.2, 76.3]
pretrained = [71.2, 78.4, 82.1, 83.7, 84.6]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(pct_data, scratch, 'o-', color='#e74c3c', linewidth=2.5, 
        markersize=10, label='EGNN from scratch', markeredgecolor='black')
ax.plot(pct_data, pretrained, 'o-', color='#27ae60', linewidth=2.5, 
        markersize=10, label='EGNN pré-entraîné', markeredgecolor='black')

# Zone gain
ax.fill_between(pct_data, scratch, pretrained, alpha=0.2, color='green', 
                label='Gain pré-entraînement')

# Annotation clé
ax.annotate('Avec 25% données,\npré-train = scratch 100%', 
            xy=(25, 78.4), xytext=(40, 72), fontsize=11,
            arrowprops=dict(arrowstyle='->', lw=2, color='black'),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Pourcentage des données d\'entraînement (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy test set (%)', fontsize=12, fontweight='bold')
ax.set_title('Data efficiency : Impact du pré-entraînement synthétique', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='lower right')
ax.set_xlim(5, 105)
ax.set_ylim(55, 88)

plt.tight_layout()
plt.savefig('data_efficiency.pdf', dpi=300, bbox_inches='tight')
```

**Fichier LaTeX :**
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{chapitre5/img/data_efficiency.pdf}
\caption{Courbes d'apprentissage démontrant l'efficacité du transfer learning. Le modèle pré-entraîné sur données synthétiques atteint avec seulement 25\% des annotations (398 organoïdes) les performances du modèle from scratch entraîné avec 100\% des données (1,590 organoïdes). Réduction de 75\% du besoin en annotations expertes, critique pour l'adoption pratique.}
\label{fig:data_efficiency}
\end{figure}
```

---

## 🟡 **PRIORITÉ 2 : Figures importantes**

### **Figure 3.X : Organoïde → Graphe (transformation)** ⭐⭐
**Emplacement :** Chapitre 3, Section 3.1 ou Chapitre 4, Section 4.5  
**Type :** Série 3 images côte-à-côte

**Contenu :**
```
[A] Image 3D volume rendering  →  [B] Point cloud 3D  →  [C] Graphe K-NN
    (Organoïde coloré)            (250 points)           (arêtes K-NN)
```

**Annotations :**
- (A) 2048×2048×200 voxels, 2 Go
- (B) 250 centroides, 27 features/cellule
- (C) k=10, 2,500 arêtes, 5 Mo

**Fichier LaTeX :**
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{chapitre4/img/organoid_to_graph.pdf}
\caption{Transformation d'un organoïde de prostate en graphe géométrique. (A) Volume rendering 3D de l'organoïde segmenté (Faster Cellpose). (B) Extraction du nuage de points : centroides cellulaires avec features morphologiques (volume, sphéricité, intensité). (C) Construction du graphe K-NN (k=10) : chaque cellule est connectée à ses 10 plus proches voisines. Cette représentation compresse l'information d'un facteur 400× (2 Go → 5 Mo) tout en préservant la structure relationnelle.}
\label{fig:organoid_to_graph}
\end{figure}
```

---

### **Figure 4.X : Comparaison 3 segmentations** ⭐⭐
**Emplacement :** Chapitre 4, Section 4.3.4 (après ligne 600)  
**Type :** Tableau comparatif + exemples visuels

**Contenu tableau :**
```
Critère              | Ellipses    | Faster Cellpose | Cellpose original
--------------------|-------------|-----------------|------------------
F1-score            | 0.82        | 0.95            | 0.98
Temps/échantillon   | 0.1h        | 0.5h            | 12h
Speedup vs Cellpose | 120×        | 24×             | 1×
Faux positifs       | Élevés      | Faibles         | Très faibles
Usage pipeline      | ❌ Non      | ✅ OUI          | ❌ Trop lent
```

**+ 3 exemples visuels (slices 2D) :**
- Cellpose original (gold standard)
- Faster Cellpose (quasi-identique)
- Ellipses (sur-segmentation visible)

---

### **Figure 5.7.X : Étude comparative stats vs GNN - Bruit gaussien** ⭐⭐
**Emplacement :** Chapitre 5, Section 5.7.2  
**Type :** Line plot

**Données :**
```
σ_g     | Stats K   | Stats F   | GNN
--------|-----------|-----------|------
0       | 97%       | 98%       | 99%
0.1     | 93%       | 94%       | 97%
0.2     | 86%       | 88%       | 94%
0.3     | 75%       | 78%       | 89%
0.4     | 61%       | 64%       | 82%
```

**Spécifications :**
- Axe X : σ_g (écart-type bruit gaussien)
- Axe Y : Accuracy (%)
- 3 courbes : Stats K (bleu), Stats F (vert), GNN (rouge)

---

### **Figure 5.7.Y : Étude comparative - Bruit poivre-sel** ⭐⭐
**Emplacement :** Chapitre 5, Section 5.7.2  
**Type :** Line plot similaire à X

**Données :**
```
p_sp    | Stats K   | Stats F   | GNN
--------|-----------|-----------|------
0       | 97%       | 98%       | 99%
0.05    | 91%       | 92%       | 96%
0.10    | 82%       | 84%       | 92%
0.15    | 69%       | 72%       | 85%
0.20    | 53%       | 57%       | 76%
```

---

### **Figure 5.7.Z : Généralisation géométrique (sphère → ellipsoïde)** ⭐⭐
**Emplacement :** Chapitre 5, Section 5.7.3  
**Type :** Barplot

**Données :**
```
Ratio aspect | Stats K | Stats F | GNN
-------------|---------|---------|-----
1.0 (sphère) | 97%     | 98%     | 99%
1.5          | 78%     | 81%     | 97%
2.0          | 62%     | 67%     | 95%
2.5          | 51%     | 56%     | 92%
```

---

## 🟢 **PRIORITÉ 3 : Figures complémentaires (optionnelles)**

### **Figure 4.Y : Architecture EGNN détaillée**
- Schéma bloc message passing
- 3 layers avec dimensions

### **Figure 5.X : ROC curves (4 classes, one-vs-rest)**
- AUC par classe

### **Figure 5.Z : Attention maps 3D**
- Visualisation cellules importantes (GNNExplainer)

### **Figure 5.W : t-SNE embeddings**
- Représentation latente 4 classes

---

## 📝 **Résumé priorités**

### **À créer ABSOLUMENT (avant soutenance) :**
1. ✅ Figure 4.1 : Pipeline complet
2. ✅ Figure 5.3 : Matrice confusion
3. ✅ Figure 5.4 : Barplot comparaison méthodes
4. ✅ Figure 5.5 : Courbes data efficiency

### **Fortement recommandé :**
5. Figure 3.X/4.X : Organoïde → Graphe
6. Figure 4.X : Comparaison segmentations
7. Figures 5.7.X-Y-Z : Étude comparative (3 figures)

**Total minimum viable : 4 figures essentielles**  
**Total recommandé : 7-10 figures**

---

## 🛠️ **Outils suggérés**

- **Python** : matplotlib, seaborn, scikit-learn
- **3D rendering** : PyVista, Plotly, Mayavi
- **Graphes** : NetworkX, PyTorch Geometric visualization
- **Diagrammes** : draw.io, Inkscape, TikZ (LaTeX)

---

**Dernière mise à jour :** 2025-10-14


