# Figures Essentielles - Classées par Ordre d'Importance

## 🔴 PRIORITÉ MAXIMALE (20 figures) - À produire en premier
*Ces figures sont absolument indispensables pour la compréhension et la défense de la thèse*

### 1. **Figure 1.7 : Phénotypes d'organoïdes de prostate** ⭐⭐⭐
   - **Pourquoi critique** : Définit visuellement le cœur du problème scientifique
   - Images confocales haute qualité des 4 phénotypes
   - Visualisation 3D (rendu volumique) pour chaque phénotype
   - Annotations des caractéristiques morphologiques distinctives
   - Zoom sur organisation cellulaire
   - **Impact** : Première chose que le jury doit voir pour comprendre les données

### 2. **Figure 1.6 : Pipeline complet de bout en bout** ⭐⭐⭐
   - **Pourquoi critique** : Vue d'ensemble de toute la méthodologie
   - Schéma bloc : Image 3D → Prétraitement → Segmentation → Graphe → GNN → Prédiction
   - Indication des volumes de données (Go → Mo)
   - Temps de traitement par étape
   - **Impact** : Donne la structure de la contribution technique

### 3. **Figure 5.17 : Matrice de confusion (test set réel)** ⭐⭐⭐
   - **Pourquoi critique** : Résultat principal de validation
   - Heatmap 4×4 : Chouxfleurs, Cystiques, Compact, Kératinisés
   - 89.2% de prédictions correctes
   - Analyse des confusions principales (24 cas Choux↔Cyst)
   - **Impact** : Preuve quantitative de la performance du modèle

### 4. **Figure 5.20 : Courbes d'apprentissage (data efficiency)** ⭐⭐⭐
   - **Pourquoi critique** : Démontre l'apport majeur du transfer learning
   - Accuracy vs % données (10%, 25%, 50%, 75%, 100%)
   - Gain maximal +12.9% à 10% des données
   - Réduction de 75% des annotations nécessaires
   - **Impact** : Valide la stratégie synthétiques → réels

### 5. **Figure 5.1 : Robustesse au bruit gaussien** ⭐⭐⭐
   - **Pourquoi critique** : Justifie le choix des GNN vs statistiques spatiales
   - Accuracy vs σ_g (0 à 0.8)
   - Comparaison Stats spatiales vs GNN (différentes profondeurs)
   - Identification profondeur optimale (L=5-6)
   - **Impact** : Validation de l'approche GNN dans l'étude comparative

### 6. **Figure 5.3 : Généralisation géométrique (ellipsoïdes)** ⭐⭐⭐
   - **Pourquoi critique** : Démontre la flexibilité topologique des GNN
   - Accuracy vs Rapport d'aspect (1:1 à 5:1)
   - Point d'inversion où GNN surpasse statistiques (ratio ~2.5)
   - **Impact** : Argument clé pour justifier GNN sur organoïdes irréguliers

### 7. **Figure 4.6 : Comparaison méthodes de segmentation** ⭐⭐⭐
   - **Pourquoi critique** : Valide la contribution méthodologique Faster Cellpose
   - Résultats visuels : Ellipses / StarDist / Cellpose / Faster Cellpose
   - Trade-off précision-vitesse (F1=0.95, 6s/coupe vs 30s)
   - Gain facteur 5× en temps
   - **Impact** : Justifie l'optimisation et rend le pipeline praticable

### 8. **Figure 5.8 : Régression du coefficient de clustering** ⭐⭐⭐
   - **Pourquoi critique** : Validation sur données synthétiques
   - Scatter plot Prédit vs Vrai
   - R² = 0.945 (94.5% variance capturée)
   - Densité de points par région
   - **Impact** : Prouve que le modèle apprend les patterns spatiaux

### 9. **Figure 4.11 : Processus ponctuels sur la sphère** ⭐⭐⭐
   - **Pourquoi critique** : Illustre la génération de données synthétiques
   - Visualisations 3D : Poisson homogène, Matérn faible, Matérn fort
   - Projection 2D et vue 3D
   - **Impact** : Explique visuellement le continuum synthétique

### 10. **Figure 3.10 : EGNN Architecture** ⭐⭐⭐
   - **Pourquoi critique** : Cœur technique de l'approche
   - Schéma détaillé d'une couche EGNN
   - Messages invariants (distances), mise à jour équivariante coordonnées
   - Flux d'information clairement identifiés
   - **Impact** : Compréhension de l'architecture équivariante

### 11. **Figure 5.16 : Accuracy 5-fold CV (réels)** ⭐⭐
   - **Pourquoi critique** : Comparaison de toutes les approches
   - Boxplots : EGNN pre-trained (84.6%), from scratch (76.3%), CNN 3D (81.2%), RF (72.4%)
   - Points des 5 folds
   - **Impact** : Validation statistique robuste

### 12. **Figure 5.24 : Cellules importantes (GNNExplainer)** ⭐⭐
   - **Pourquoi critique** : Démontre l'interprétabilité
   - 6 organoïdes (3 Chouxfleurs, 3 Cystiques)
   - Heat map d'importance (rouge=important, bleu=peu important)
   - Annotations des régions discriminantes
   - **Impact** : Validation biologique des prédictions

### 13. **Figure 1.5 : Notre dataset OrganoProstate-2K** ⭐⭐
   - **Pourquoi critique** : Caractérisation des données
   - Composition (Chouxfleurs 61.8%, Cystiques 36%, autres 2.2%)
   - Timeline de collecte (Mai 2023 - Février 2025)
   - Exemples d'images par phénotype
   - **Impact** : Contexte expérimental complet

### 14. **Figure 4.8 : Du nuage de points au graphe** ⭐⭐
   - **Pourquoi critique** : Abstraction clé de l'approche
   - Étapes visuelles : Points → K-NN → Symétrisation → Graphe final
   - **Impact** : Illustre la transformation image→graphe

### 15. **Figure 5.10 : Comparaison architectures GNN** ⭐⭐
   - **Pourquoi critique** : Justifie le choix EGNN
   - Barres : MSE et R² pour GCN, GAT, EGNN
   - EGNN : MSE -55% vs GCN, -43% vs GAT
   - **Impact** : Validation empirique du choix architectural

### 16. **Figure 4.14 : Validation statistique des synthétiques** ⭐⭐
   - **Pourquoi critique** : Prouve le réalisme des données synthétiques
   - Fonctions de Ripley K, F, G : théoriques vs observées
   - Enveloppes de confiance Monte Carlo
   - Superposition avec données réelles
   - **Impact** : Rigueur scientifique de la génération

### 17. **Figure 5.29 : Triangle précision-interprétabilité-efficacité** ⭐⭐
   - **Pourquoi critique** : Synthèse du positionnement
   - Diagramme ternaire avec toutes les approches
   - GNN : compromis optimal
   - **Impact** : Vision synthétique pour la discussion

### 18. **Figure 3.4 : Message Passing Neural Network** ⭐⭐
   - **Pourquoi critique** : Fondement théorique des GNN
   - Mécanisme sur 3 itérations
   - Nœud central collectant info des voisins
   - h^(0) → h^(1) → h^(2)
   - **Impact** : Pédagogie pour comprendre les GNN

### 19. **Figure 4.1 : Architecture du pipeline (détaillée)** ⭐⭐
   - **Pourquoi critique** : Version technique complète
   - Tous les paramètres, dimensions exactes
   - Temps et ressources par étape
   - **Impact** : Reproductibilité technique

### 20. **Figure 5.4 : Synthèse comparative GNN vs Stats** ⭐⭐
   - **Pourquoi critique** : Conclusion de l'étude comparative
   - Radar multi-critères : bruit, généralisation, flexibilité, interprétabilité
   - **Impact** : Vision holistique de la comparaison

---

## 🟠 HAUTE PRIORITÉ (25 figures) - À produire ensuite
*Figures importantes pour la compréhension approfondie et la validation*

### 21. **Figure 5.18 : F1-scores par classe**
   - Barres groupées : Précision, Rappel, F1 pour chaque phénotype
   - Indication du support (nombre d'exemples)
   - **Utilité** : Détail des performances par classe

### 22. **Figure 4.12 : Continuum Poisson-Matérn**
   - Série montrant transition graduelle
   - Coefficient clustering 0 → 1
   - 10-15 exemples espacés
   - **Utilité** : Visualise le spectre complet des synthétiques

### 23. **Figure 5.12 : Impact des features géométriques (ablation)**
   - Barres MSE : Complet / Sans 3D / Sans volume / 2D seul
   - Dégradations relatives (MSE ×4 sans 3D)
   - **Utilité** : Importance de chaque composante

### 24. **Figure 3.9 : Équivariance vs Invariance**
   - Illustration conceptuelle objet original → rotation
   - Output invariant (scalaire) vs équivariant (vecteur transformé)
   - **Utilité** : Concept fondamental EGNN

### 25. **Figure 5.13 : Influence stratégie de connectivité**
   - MSE pour K-NN (k=5,10,15,20), Rayon, Delaunay
   - Temps de construction
   - Optimal : k=10
   - **Utilité** : Justifie le choix de construction du graphe

### 26. **Figure 4.4 : Cellpose - Principe**
   - Champs de gradients
   - Flow tracking regroupant pixels
   - Exemple sur coupe réelle
   - **Utilité** : Comprendre la segmentation state-of-the-art

### 27. **Figure 2.4 : Comparaison méthodes de segmentation**
   - Images : Original → Watershed → StarDist → Cellpose
   - Tableau performance
   - **Utilité** : Contexte état de l'art segmentation

### 28. **Figure 5.19 : Courbes ROC multiclasses**
   - 4 courbes ROC (one-vs-rest)
   - AUC par classe
   - **Utilité** : Évaluation probabiliste

### 29. **Figure 4.13 : Tessellation de Voronoï sphérique**
   - Points → Voronoï → Graphe dual
   - Zoom sur cellules individuelles
   - **Utilité** : Méthode de construction du graphe synthétique

### 30. **Figure 5.15 : Sensibilité hyperparamètres**
   - Panel 4 sous-graphiques : Couches / Dimension / LR / Dropout
   - **Utilité** : Guide pour réglage des paramètres

### 31. **Figure 1.1 : Types d'organoïdes et applications**
   - Schéma : intestin, cerveau, foie, poumon, prostate
   - Images microscopiques + applications
   - **Utilité** : Contexte biologique large

### 32. **Figure 5.21 : Convergence accélérée**
   - Loss vs époques : from scratch (80) vs pre-trained (30)
   - **Utilité** : Avantage du transfer learning

### 33. **Figure 5.22 : Visualisation embeddings (t-SNE)**
   - Panel : from scratch (clusters partiels) vs pre-trained (séparés)
   - **Utilité** : Qualité de l'espace latent appris

### 34. **Figure 4.15 : Comparaison synthétiques vs réels**
   - PCA, t-SNE des embeddings
   - Distributions métriques topologiques (boxplots)
   - **Utilité** : Validation de la couverture de l'espace phénotypique

### 35. **Figure 5.26 : Motifs topologiques discriminants**
   - Sous-graphes fréquents par classe
   - Matérn : triangles, cliques / Poisson : régularité
   - **Utilité** : Patterns biologiques capturés

### 36. **Figure 3.11 : Panorama GNN géométriques**
   - Arbre taxonomique des architectures
   - Positionnement SchNet, DimeNet, PaiNN, EGNN, NequIP
   - **Utilité** : Contexte théorique des GNN géométriques

### 37. **Figure 4.5 : Faster Cellpose - Optimisations**
   - Knowledge Distillation (Teacher → Student)
   - Architecture allégée, pruning 30%
   - Graphique temps vs performance
   - **Utilité** : Détail de la contribution méthodologique

### 38. **Figure 5.23 : Généralisation cross-site**
   - Barres : Test Paris, Test Nice
   - Drop de performance (-5.4%)
   - **Utilité** : Robustesse inter-laboratoires

### 39. **Figure 4.10 : Stratégies de connectivité**
   - Comparaison visuelle : K-NN (k=5,10,15), Rayon, Hybride
   - Sur même organoïde
   - **Utilité** : Options de construction de graphe

### 40. **Figure 5.28 : Comparaison efficacité computationnelle**
   - Tableau : Mémoire GPU / Temps/org / Throughput
   - Manuel, CNN 3D, GNN
   - **Utilité** : Avantage pratique de l'approche

### 41. **Figure 3.5 : Architectures GNN standards**
   - Schémas comparatifs : GCN, GAT, GraphSAGE, GIN
   - **Utilité** : Contexte des baselines

### 42. **Figure 4.16 : EGNN détaillé pour organoïdes**
   - Architecture complète adaptée
   - Dimensions : 4D → 256 → ... → 2
   - Residual connections, LayerNorm
   - **Utilité** : Détails d'implémentation

### 43. **Figure 5.11 : Courbes d'apprentissage (synthétiques)**
   - Train/Val loss vs époques pour GCN, GAT, EGNN
   - Gap train-val
   - **Utilité** : Comportement d'entraînement

### 44. **Figure 1.4 : Défis de l'analyse d'organoïdes**
   - Panel 4 images : Variabilité / Complexité 3D / Hétérogénéité / Contraintes
   - **Utilité** : Motivation du problème

### 45. **Figure 4.7 : Clustering DBSCAN**
   - Nuage 3D avec plusieurs organoïdes
   - Résultat clustering (couleurs)
   - **Utilité** : Séparation des organoïdes individuels

---

## 🟡 PRIORITÉ MOYENNE (20 figures) - Utiles pour la complétude
*Figures renforçant la compréhension mais moins critiques*

### 46. **Figure 5.14 : Équivariance E(3) - Test rotations**
   - MSE avec/sans rotations pour GAT vs EGNN
   - Démonstration robustesse
   - **Utilité** : Validation expérimentale équivariance

### 47. **Figure 5.9 : Distribution des erreurs**
   - Histogramme erreur de prédiction
   - Ajustement gaussien, MAE, médiane
   - **Utilité** : Analyse fine des erreurs

### 48. **Figure 4.9 : Features des nœuds et arêtes**
   - Illustration cellule (position 3D, volume)
   - Arête (distance, vecteur)
   - **Utilité** : Définition formelle des features

### 49. **Figure 5.27 : Features discriminantes (SHAP)**
   - Beeswarm plot SHAP values
   - Top 10 features
   - **Utilité** : Importance relative des features

### 50. **Figure 3.1 : Représentations de graphes**
   - Graphe visuel + Matrice adjacence + Liste + COO
   - **Utilité** : Pédagogie fondements graphes

### 51. **Figure 2.1 : Mécanismes d'auto-organisation**
   - Interactions cellule-cellule, cellule-matrice
   - Gradients morphogénétiques
   - **Utilité** : Biologie fondamentale

### 52. **Figure 3.6 : Mécanisme d'attention (GAT)**
   - Calcul coefficients α_ij
   - Heat map attention sur graphe
   - **Utilité** : Détail technique GAT

### 53. **Figure 5.25 : Poids d'attention (GAT)**
   - Coefficients α_ij visualisés
   - Épaisseur arêtes proportionnelle
   - **Utilité** : Interprétabilité GAT

### 54. **Figure 1.2 : Comparaison modèles biologiques**
   - 2D vs organoïdes vs animaux
   - Radar plot ou barres
   - **Utilité** : Positionnement des organoïdes

### 55. **Figure 4.17 : Composants EGNN**
   - Décomposition couche : φ_e, φ_x, φ_h
   - **Utilité** : Détails des opérations EGNN

### 56. **Figure 3.2 : Graphes géométriques vs abstraits**
   - Graphe social vs nuage points 3D
   - **Utilité** : Distinction conceptuelle

### 57. **Figure 5.5 : Fonctions Ripley - Validation théorique**
   - K, F, G : théoriques vs simulés
   - Enveloppes 95%
   - **Utilité** : Validation statistique rigoureuse

### 58. **Figure 5.6 : Comparaison topologique synthétiques/réels**
   - Histogrammes degré, clustering, diamètre
   - Tests KS avec p-values
   - **Utilité** : Similarité topologique

### 59. **Figure 3.3 : Construction graphes géométriques**
   - K-NN, ε-radius, Delaunay sur même nuage
   - **Utilité** : Options de construction

### 60. **Figure 2.2 : Modalités d'imagerie 3D**
   - Schémas confocale / light-sheet / multiphoton
   - Tableau comparatif
   - **Utilité** : Contexte technique imagerie

### 61. **Figure 5.30 : Cas d'échec et limitations**
   - Panel : erreurs de segmentation, phénotypes ambigus, très denses
   - **Utilité** : Honnêteté scientifique

### 62. **Figure 1.3 : Formation organoïde au fil du temps**
   - Série temporelle J0→J10+
   - Annotation des processus
   - **Utilité** : Biologie développementale

### 63. **Figure 5.7 : Distributions features cellulaires**
   - Boxplots synthétiques vs réels : Volume, distance, degré
   - p-values KS
   - **Utilité** : Réalisme morphologique

### 64. **Figure 3.7 : Architecture DeepSets**
   - Schéma φ → AGG → ρ
   - Invariance permutations
   - **Utilité** : Contexte théorique (baseline conceptuelle)

### 65. **Figure 4.2 : Normalisation d'intensité**
   - Histogrammes avant/après
   - Images comparatives
   - **Utilité** : Prétraitement technique

---

## 🟢 PRIORITÉ BASSE (17 figures) - Compléments et annexes
*Figures optionnelles renforçant certains points spécifiques*

### 66. **Figure 5.2 : Robustesse bruit poivre et sel**
   - Similar à 5.1 mais pour bruit structurel
   - **Utilité** : Complément robustesse

### 67. **Figure 2.5 : Limitations approches existantes**
   - Matrice avantages/inconvénients
   - **Utilité** : Synthèse état de l'art

### 68. **Figure 2.3 : Défis d'imagerie**
   - Artefacts, atténuation, aberrations
   - **Utilité** : Défis techniques

### 69. **Figure 3.8 : PointNet**
   - Architecture avec T-Net
   - Limitations
   - **Utilité** : Alternative conceptuelle

### 70. **Figure 3.12 : Spectre expressivité géométrique**
   - SchNet → DimeNet → PaiNN → NequIP
   - Trade-off complexité/expressivité
   - **Utilité** : Vision continue des approches

### 71. **Figure 4.3 : Débruitage**
   - Comparaison filtres : médian, gaussien, combiné
   - SNR avant/après
   - **Utilité** : Détail prétraitement

### 72. **Figure 2.6 : GNN en histopathologie**
   - Graphes cellulaires en pathologie 2D
   - Extension 3D
   - **Utilité** : Applications connexes

### 73. **Figure 6.1 : Feuille de route méthodologique**
   - Timeline améliorations : court/moyen/long terme
   - **Utilité** : Perspectives

### 74. **Figure 6.2 : Vision intégrative multi-modale**
   - Schéma : Imagerie + Transcripto + Protéo + Clinique
   - **Utilité** : Vision future

### 75. **Figure 5.31-82 : Figures annexes techniques**
   - Architectures CNN, régularisation, processus ponctuels détaillés
   - Diagrammes UML, optimisations, protocoles annotation
   - **Utilité** : Compléments techniques pour experts

---

## 📊 RÉSUMÉ QUANTITATIF

| Priorité | Nombre | % | Production |
|----------|--------|---|------------|
| 🔴 Maximale | 20 | 24% | Semaine 1-2 |
| 🟠 Haute | 25 | 31% | Semaine 3-4 |
| 🟡 Moyenne | 20 | 24% | Semaine 5-6 |
| 🟢 Basse | 17 | 21% | Optionnel |
| **TOTAL** | **82** | **100%** | **6 semaines** |

---

## 🎯 STRATÉGIE DE PRODUCTION RECOMMANDÉE

### Phase 1 (URGENT - 1 semaine) : Les 10 figures absolument critiques
1. Figure 1.7 (Phénotypes) ← **COMMENCE ICI**
2. Figure 1.6 (Pipeline)
3. Figure 5.17 (Matrice confusion)
4. Figure 5.20 (Data efficiency)
5. Figure 5.1 (Robustesse bruit)
6. Figure 5.3 (Généralisation géométrique)
7. Figure 4.6 (Comparaison segmentation)
8. Figure 5.8 (Régression clustering)
9. Figure 4.11 (Processus ponctuels)
10. Figure 3.10 (EGNN)

**→ Avec ces 10 figures, vous pouvez déjà défendre 80% de votre thèse**

### Phase 2 (Important - 1 semaine) : Les 10 figures suivantes
11-20 de la liste Priorité Maximale

### Phase 3 (Consolidation - 2 semaines) : Haute priorité
Figures 21-45

### Phase 4 (Finition - 2 semaines) : Reste si temps disponible
Figures 46-82

---

## 💡 CONSEILS PRATIQUES

1. **Parallélisez** : 
   - Données réelles (Fig 1.7, 5.17) : demander au labo
   - Graphiques synthétiques (Fig 5.8, 5.1) : scripts Python
   - Schémas conceptuels (Fig 1.6, 3.10) : BioRender/Inkscape

2. **Réutilisez** :
   - Mêmes organoïdes pour Fig 1.7, 4.8, 5.24
   - Même style graphique pour toutes les courbes

3. **Automatisez** :
   - Scripts Python pour toutes les figures de résultats
   - Templates LaTeX/TikZ pour schémas répétitifs

4. **Déléguez si possible** :
   - Collaborateur pour schémas BioRender
   - Labo pour acquisitions microscopie haute qualité

