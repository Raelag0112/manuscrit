# Figures Essentielles pour la Thèse

## Introduction (Chapitre 1)

### Contexte Biologique
1. **Figure 1.1 : Types d'organoïdes et leurs applications**
   - Schéma illustrant différents types d'organoïdes (intestin, cerveau, foie, poumon, prostate)
   - Avec images microscopiques représentatives de chaque type
   - Légende indiquant les applications (recherche fondamentale, criblage, médecine personnalisée)

2. **Figure 1.2 : Comparaison des modèles biologiques**
   - Tableau comparatif visuel : cultures 2D vs organoïdes 3D vs modèles animaux
   - Axes : complexité, pertinence physiologique, coût, débit, contrôle expérimental
   - Radar plot ou diagramme en barres

3. **Figure 1.3 : Formation d'un organoïde au fil du temps**
   - Série temporelle montrant les étapes de développement (J0 à J10+)
   - Images confocales à différents temps
   - Annotation des processus clés (agrégation, polarisation, formation lumen, bourgeonnement)

### Problématique
4. **Figure 1.4 : Défis de l'analyse d'organoïdes**
   - Panel de 4 images illustrant les défis :
     a) Variabilité morphologique (taille, forme)
     b) Complexité 3D (coupes à différentes profondeurs)
     c) Hétérogénéité cellulaire (types cellulaires différents)
     d) Contraintes computationnelles (volume de données)

5. **Figure 1.5 : Notre dataset OrganoProstate-2K**
   - Composition du dataset (diagramme circulaire ou barres empilées)
   - Distribution des phénotypes (Chouxfleurs 61.8%, Cystiques 36%, Compact 1.8%, Kératinisés 0.4%)
   - Timeline de collecte (Mai 2023 - Février 2025)
   - Exemples d'images pour chaque phénotype

### Contributions
6. **Figure 1.6 : Pipeline complet de bout en bout**
   - Schéma bloc du pipeline entier :
     * Image 3D brute → Prétraitement → Segmentation (Faster Cellpose)
     * → Extraction features → Clustering DBSCAN → Construction graphes
     * → Classification GNN → Prédiction + Interprétabilité
   - Indication des volumes de données à chaque étape (Go → Mo)
   - Temps de traitement par étape

7. **Figure 1.7 : Phénotypes d'organoïdes de prostate**
   - Images confocales haute qualité des 4 phénotypes
   - Visualisation 3D (rendu volumique) pour chaque phénotype
   - Annotations des caractéristiques morphologiques distinctives
   - Zoom sur organisation cellulaire pour chaque type

## État de l'Art (Chapitre 2)

### Biologie des Organoïdes
8. **Figure 2.1 : Mécanismes d'auto-organisation**
   - Schéma des interactions cellule-cellule (jonctions adhérentes, gap junctions)
   - Schéma des interactions cellule-matrice (intégrines, ECM)
   - Gradients de signalisation morphogénétique (Wnt, BMP, FGF)

### Imagerie
9. **Figure 2.2 : Modalités d'imagerie 3D**
   - Comparaison schématique confocale / light-sheet / multiphoton
   - Principe optique de chaque modalité
   - Tableau comparatif (résolution, profondeur, vitesse, photoblanchiment)

10. **Figure 2.3 : Défis d'imagerie d'organoïdes**
    - Illustration des artefacts : diffusion de lumière, atténuation avec la profondeur
    - Graphique intensité vs profondeur Z
    - Exemples d'aberrations optiques

### Méthodes d'Analyse
11. **Figure 2.4 : Comparaison des méthodes de segmentation**
    - Images comparatives : Original → Watershed → StarDist → Cellpose
    - Tableau performance (F1-score, Temps, Avantages/Inconvénients)
    - Focus sur cas difficiles (cellules collées, morphologies irrégulières)

12. **Figure 2.5 : Limitations des approches existantes**
    - Schéma conceptuel :
      * Analyse manuelle (temps, subjectivité, non-scalable)
      * CNN 3D (mémoire, downsampling, perte d'information)
      * Descripteurs manuels (expressivité limitée, engineering)
    - Matrice avantages/inconvénients

13. **Figure 2.6 : GNN en histopathologie**
    - Exemples de graphes cellulaires en pathologie 2D
    - Visualisation d'un tissu tumoral comme graphe (nœuds=cellules, arêtes=voisinage)
    - Extension conceptuelle à la 3D pour les organoïdes

## Fondements Théoriques (Chapitre 3)

### Théorie des Graphes
14. **Figure 3.1 : Représentations de graphes**
    - Exemple de graphe simple avec représentations multiples :
      * Visualisation visuelle du graphe
      * Matrice d'adjacence A
      * Liste d'adjacence
      * Edge list (format COO)

15. **Figure 3.2 : Graphes géométriques vs abstraits**
    - Comparaison côte à côte :
      * Graphe social (abstrait, topologie seule)
      * Nuage de points 3D / organoïde (géométrique, coordonnées spatiales)
    - Mise en évidence des différences (invariances, transformations)

16. **Figure 3.3 : Construction de graphes géométriques**
    - Illustration des stratégies sur un même nuage de points :
      * K-NN (k=5)
      * ε-radius
      * Triangulation de Delaunay
    - Comparaison visuelle des connectivités résultantes

### GNN Standards
17. **Figure 3.4 : Message Passing Neural Network (schéma général)**
    - Illustration du mécanisme de message passing sur 3 itérations
    - Nœud central collectant information de ses voisins
    - Mise à jour progressive des représentations h^(0) → h^(1) → h^(2)
    - Formules mathématiques superposées

18. **Figure 3.5 : Architectures GNN standards**
    - Schémas comparatifs de 4 architectures :
      * GCN (agrégation normalisée)
      * GAT (attention multi-têtes)
      * GraphSAGE (sampling)
      * GIN (agrégation somme + MLP)
    - Focus sur les différences clés de chaque approche

19. **Figure 3.6 : Mécanisme d'attention (GAT)**
    - Diagramme détaillé du calcul des coefficients d'attention α_ij
    - Visualisation des poids d'attention sur un exemple de graphe
    - Heat map montrant quels voisins reçoivent plus d'attention

### DeepSets et PointNet
20. **Figure 3.7 : Architecture DeepSets**
    - Schéma φ → AGG → ρ
    - Illustration de l'invariance aux permutations
    - Comparaison avec GNN (agrégation globale vs locale)

21. **Figure 3.8 : PointNet pour nuages de points**
    - Architecture complète avec T-Net
    - Application à un nuage de points 3D
    - Limitations (pas de contexte local)

### GNN Géométriques
22. **Figure 3.9 : Équivariance vs Invariance**
    - Illustration conceptuelle sur un exemple :
      * Objet original
      * Objet après rotation
      * Output invariant (scalaire identique)
      * Output équivariant (vecteur transformé)
    - Définitions mathématiques

23. **Figure 3.10 : EGNN Architecture**
    - Schéma détaillé d'une couche EGNN :
      * Messages invariants (distances)
      * Mise à jour équivariante des coordonnées
      * Mise à jour des features scalaires
    - Illustration des flots d'information

24. **Figure 3.11 : Panorama GNN géométriques**
    - Arbre taxonomique des architectures géométriques
    - Axes : Invariance/Équivariance, Représentation (scalaires/vecteurs/tenseurs), Information géométrique
    - Positionnement de SchNet, DimeNet, PaiNN, EGNN, NequIP

25. **Figure 3.12 : Spectre d'expressivité géométrique**
    - Ligne continue de complexité croissante :
      * Distances seules (SchNet)
      * Distances + Angles (DimeNet)
      * Distances + Vecteurs (PaiNN, EGNN)
      * Harmoniques sphériques (NequIP)
    - Trade-off complexité vs expressivité

## Méthodologie (Chapitre 4)

### Pipeline
26. **Figure 4.1 : Architecture du pipeline (détaillée)**
    - Version détaillée de la Figure 1.6 avec tous les paramètres
    - Flux de données avec dimensions exactes
    - Temps de calcul et ressources à chaque étape
    - Points de décision et branches optionnelles

### Prétraitement
27. **Figure 4.2 : Normalisation d'intensité**
    - Histogrammes avant/après normalisation
    - Effet de la correction de fond
    - Images comparatives (brute vs normalisée)

28. **Figure 4.3 : Débruitage**
    - Comparaison Originale → Filtre médian → Filtre gaussien → Combiné
    - SNR avant/après
    - Zoom sur détails préservés

### Segmentation
29. **Figure 4.4 : Cellpose - Principe**
    - Illustration des champs de gradients
    - Flow tracking regroupant pixels en cellules
    - Exemple sur une coupe réelle

30. **Figure 4.5 : Faster Cellpose - Optimisations**
    - Diagramme de Knowledge Distillation (Teacher → Student)
    - Architecture allégée (canaux réduits)
    - Pruning 30% des poids
    - Graphique temps vs performance

31. **Figure 4.6 : Comparaison méthodes de segmentation**
    - Résultats visuels sur même image : Ellipses / StarDist / Cellpose / Faster Cellpose
    - Tableau récapitulatif (F1-score, Temps/coupe, Temps total 1000 org)
    - Trade-off précision-vitesse (scatter plot)

### Extraction et Séparation
32. **Figure 4.7 : Clustering DBSCAN**
    - Nuage de points 3D avec plusieurs organoïdes
    - Résultat du clustering (couleurs par cluster)
    - Paramètres eps et min_samples illustrés

### Construction de Graphes
33. **Figure 4.8 : Du nuage de points au graphe**
    - Étapes visuelles :
      * Nuage de points (centroïdes cellulaires)
      * Construction K-NN (k=10)
      * Symétrisation
      * Filtrage distance max
      * Graphe final avec arêtes

34. **Figure 4.9 : Features des nœuds et arêtes**
    - Illustration d'une cellule avec ses features (position 3D, volume)
    - Illustration d'une arête avec ses features (distance, vecteur directionnel)
    - Format de données (tenseur PyG)

35. **Figure 4.10 : Stratégies de connectivité**
    - Comparaison sur même organoïde :
      * K-NN (k=5, 10, 15)
      * Rayon fixe (r=30, 50 μm)
      * Hybride (K-NN + rayon max)
    - Métriques de graphe résultantes (degré moyen, clustering)

### Données Synthétiques
36. **Figure 4.11 : Processus ponctuels sur la sphère**
    - Visualisations 3D de distributions synthétiques :
      * Poisson homogène (aléatoire complet)
      * Matérn clustering faible
      * Matérn clustering fort
    - Projection 2D et vue 3D

37. **Figure 4.12 : Continuum Poisson-Matérn**
    - Série d'images montrant la transition graduelle
    - Coefficient de clustering de 0 à 1
    - 10-15 exemples uniformément espacés

38. **Figure 4.13 : Tessellation de Voronoï sphérique**
    - Illustration du processus :
      * Points sur sphère
      * Diagramme de Voronoï résultant
      * Graphe dual (connectivité)
    - Zoom sur cellules de Voronoï individuelles

39. **Figure 4.14 : Validation statistique des synthétiques**
    - Fonctions de Ripley K, F, G :
      * Courbes théoriques vs observées
      * Enveloppes de confiance (Monte Carlo)
      * Pour Poisson et Matérn
    - Superposition avec données réelles

40. **Figure 4.15 : Comparaison synthétiques vs réels**
    - PCA : projection 2D des embeddings (synthétiques + réels)
    - t-SNE : visualisation de la séparation des classes
    - Distributions de métriques topologiques (boxplots)

### Architectures
41. **Figure 4.16 : EGNN détaillé pour organoïdes**
    - Architecture complète adaptée à notre problème
    - Dimensions de couches (input 4D → 256 → 256 → ... → 2 ou 1)
    - Residual connections, LayerNorm
    - Pooling (mean + max concatenated)

42. **Figure 4.17 : Composants EGNN**
    - Décomposition d'une couche EGNN :
      * Calcul messages (φ_e)
      * Mise à jour coordonnées (φ_x)
      * Mise à jour features (φ_h)
    - Flèches montrant les flux d'information

## Résultats (Chapitre 5)

### Étude Comparative GNN vs Statistiques Spatiales
43. **Figure 5.1 : Robustesse au bruit gaussien**
    - Graphique Accuracy vs σ_g (0 à 0.8)
    - Courbes pour : Statistiques spatiales, GNN L=2, L=3, L=4, L=5, L=6, L=7, L=8
    - Barres d'erreur (5 seeds)
    - Zone optimale de profondeur mise en évidence

44. **Figure 5.2 : Robustesse au bruit poivre et sel**
    - Graphique Accuracy vs σ_ps (0 à 0.4)
    - Même structure que Figure 5.1
    - Comparaison des dégradations

45. **Figure 5.3 : Généralisation géométrique (ellipsoïdes)**
    - Graphique Accuracy vs Rapport d'aspect (1:1 à 5:1)
    - Comparaison Statistiques spatiales vs GNN
    - Point d'inversion où GNN surpasse statistiques (ratio ~2.5)
    - Exemples visuels d'ellipsoïdes testés

46. **Figure 5.4 : Synthèse comparative**
    - Tableau radar multi-critères :
      * Robustesse bruit
      * Généralisation géométrique
      * Flexibilité topologique
      * Interprétabilité
      * Data efficiency
      * Vitesse calcul
    - Pour chaque approche (Stats spatiales, GNN, CNN 3D)

### Validation Synthétiques
47. **Figure 5.5 : Fonctions de Ripley - Validation théorique**
    - Panel de 3 graphiques (K, F, G)
    - Courbes théoriques vs simulées
    - Enveloppes de confiance 95%
    - Pour Poisson et Matérn

48. **Figure 5.6 : Comparaison topologique synthétiques/réels**
    - Histogrammes superposés :
      * Degré moyen
      * Coefficient de clustering
      * Diamètre
    - Tests KS avec p-values affichées

49. **Figure 5.7 : Distributions de features cellulaires**
    - Boxplots comparant synthétiques vs réels :
      * Volume cellulaire
      * Distance inter-cellulaire
      * Degré des nœuds
    - p-values des tests KS

### Performances Synthétiques
50. **Figure 5.8 : Régression du coefficient de clustering**
    - Scatter plot : Prédit vs Vrai
    - Ligne diagonale identité
    - Densité de points (heatmap)
    - R² = 0.945 affiché
    - Coloration par plage de clustering (faible/modéré/fort)

51. **Figure 5.9 : Distribution des erreurs**
    - Histogramme de l'erreur de prédiction
    - Ajustement gaussien superposé
    - Statistiques (MAE, médiane, 90ème percentile)

52. **Figure 5.10 : Comparaison architectures GNN**
    - Barres groupées : MSE et R² pour GCN, GAT, EGNN
    - Barres d'erreur (CV 5-fold)
    - Indication du nombre de paramètres

53. **Figure 5.11 : Courbes d'apprentissage (synthétiques)**
    - Train loss et Val loss vs époques
    - Pour GCN, GAT, EGNN
    - Identification des points de convergence
    - Gap train-val

### Ablations
54. **Figure 5.12 : Impact des features géométriques**
    - Barres comparant MSE :
      * Complet (3D + volume)
      * Sans positions 3D
      * Sans volume
      * Positions 2D seules
    - Dégradations relatives affichées

55. **Figure 5.13 : Influence de la stratégie de connectivité**
    - Graphique double axe :
      * MSE (barres) pour différentes stratégies
      * Temps de construction (ligne)
    - K-NN (k=5, 10, 15, 20), Rayon, Delaunay

56. **Figure 5.14 : Équivariance E(3)**
    - Test sur rotations aléatoires :
      * MSE sans rotation (baseline)
      * MSE avec rotations pour GAT (non-équivariant)
      * MSE avec rotations pour EGNN (équivariant)
    - Démonstration de la robustesse

57. **Figure 5.15 : Sensibilité aux hyperparamètres**
    - Panel de 4 sous-graphiques :
      * MSE vs Nombre de couches
      * MSE vs Dimension cachée
      * MSE vs Learning rate
      * MSE vs Dropout

### Performances Réelles
58. **Figure 5.16 : Accuracy 5-fold CV (réels)**
    - Boxplots pour :
      * EGNN pre-trained
      * EGNN from scratch
      * CNN 3D
      * Random Forest
    - Points individuels des 5 folds
    - Lignes de connexion

59. **Figure 5.17 : Matrice de confusion (test set réel)**
    - Heatmap 4×4 : Chouxfleurs, Cystiques, Compact, Kératinisés
    - Valeurs absolues dans cellules
    - Normalisation par ligne (pourcentages)
    - Colormap divergente

60. **Figure 5.18 : F1-scores par classe**
    - Barres groupées : Précision, Rappel, F1
    - Pour chaque phénotype
    - Indication du support (nombre d'exemples)

61. **Figure 5.19 : Courbes ROC multiclasses**
    - 4 courbes ROC (one-vs-rest)
    - AUC affichée pour chaque classe
    - Ligne diagonale (hasard)

### Transfer Learning
62. **Figure 5.20 : Courbes d'apprentissage (data efficiency)**
    - Graphique Accuracy vs % données (10%, 25%, 50%, 75%, 100%)
    - 2 courbes : EGNN from scratch, EGNN pre-trained
    - Zone de gain maximal mise en évidence
    - Réduction d'annotations nécessaires annotée

63. **Figure 5.21 : Convergence accélérée**
    - Loss vs époques
    - Comparaison from scratch (80 époques) vs pre-trained (30 époques)
    - Marqueurs de convergence

64. **Figure 5.22 : Visualisation embeddings (t-SNE)**
    - Panel 2×1 :
      * From scratch : clusters partiels
      * Pre-trained : clusters bien séparés
    - Coloration par phénotype
    - Variance expliquée

### Généralisation Inter-Expérimentale
65. **Figure 5.23 : Généralisation cross-site**
    - Barres groupées : Test Paris, Test Nice
    - Pour from scratch et pre-trained
    - Drop de performance affiché
    - Indication des différences expérimentales

### Interprétabilité
66. **Figure 5.24 : Cellules importantes (GNNExplainer)**
    - Visualisations 3D de 6 organoïdes (3 Chouxfleurs, 3 Cystiques)
    - Heat map d'importance superposée (rouge=important, bleu=peu important)
    - Annotations des régions discriminantes

67. **Figure 5.25 : Poids d'attention (GAT)**
    - Visualisation des coefficients α_ij sur graphe 2D projeté
    - Épaisseur des arêtes proportionnelle à l'attention
    - Focus sur nœud central et son voisinage

68. **Figure 5.26 : Motifs topologiques discriminants**
    - Sous-graphes fréquents dans chaque classe :
      * Matérn : triangles fermés, cliques
      * Poisson : structures régulières
    - Fréquence d'apparition (barres)

69. **Figure 5.27 : Features discriminantes (SHAP)**
    - Beeswarm plot des SHAP values
    - Top 10 features les plus importantes
    - Impact positif/négatif pour chaque classe

### Discussion
70. **Figure 5.28 : Comparaison efficacité computationnelle**
    - Tableau visual :
      * Mémoire GPU
      * Temps/organoïde
      * Throughput (org/min)
    - Pour Manuel, CNN 3D, GNN (ours)

71. **Figure 5.29 : Triangle précision-interprétabilité-efficacité**
    - Diagramme triangulaire (ternaire)
    - Positionnement de chaque approche :
      * Manuel
      * Stats spatiales
      * CNN 3D
      * GNN (ours)
      * Random Forest

72. **Figure 5.30 : Cas d'échec et limitations**
    - Panel montrant exemples d'erreurs :
      * Segmentation incorrecte → erreur de graphe
      * Phénotype intermédiaire ambigu
      * Organoïde très dense (>1500 cellules)
    - Annotations des causes d'erreur

## Conclusion (Chapitre 6)

### Perspectives
73. **Figure 6.1 : Feuille de route méthodologique**
    - Timeline des améliorations proposées :
      * Court terme (6-12 mois)
      * Moyen terme (1-2 ans)
      * Long terme (2-5 ans)
    - Extensions GNN, validation clinique, analyse spatio-temporelle

74. **Figure 6.2 : Vision intégrative multi-modale**
    - Schéma conceptuel intégrant :
      * Imagerie 3D (notre approche actuelle)
      * Transcriptomique (spatial RNA-seq)
      * Protéomique (CyTOF, IMC)
      * Données cliniques (réponse patient)
    - GNN comme architecture unificatrice

## Annexes

### Annexe A : Deep Learning
75. **Figure A.1 : Architectures CNN classiques**
    - Évolution historique : LeNet → AlexNet → VGG → ResNet → EfficientNet
    - Nombre de paramètres et performances ImageNet

76. **Figure A.2 : Techniques de régularisation**
    - Illustration visuelle :
      * Dropout (neurones désactivés)
      * Batch Normalization (distributions)
      * Data Augmentation (transformations)
      * Weight Decay (courbes de loss)

### Annexe C : Processus Ponctuels
77. **Figure C.1 : Types de processus ponctuels**
    - Comparaison visuelle :
      * Poisson homogène (CSR)
      * Poisson inhomogène
      * Matérn clustering
      * Strauss (répulsion)
    - Fonctions K correspondantes

78. **Figure C.2 : Estimation fonction K de Ripley**
    - Dérivation visuelle de la formule
    - Correction de bord illustrée
    - Enveloppes Monte Carlo

### Annexe D : Implémentation
79. **Figure D.1 : Architecture logicielle**
    - Diagramme UML des modules principaux
    - Dépendances entre composants
    - Flux de données

80. **Figure D.2 : Optimisations GPU**
    - Comparaison temps de calcul :
      * Sequential CPU
      * Parallel CPU
      * GPU naive
      * GPU optimized (batching, mixed precision)

### Annexe E : Données
81. **Figure E.1 : Protocole d'annotation**
    - Interface utilisateur pour annotation
    - Critères de classification visuels
    - Exemples borderline nécessitant expertise

82. **Figure E.2 : Statistiques détaillées du dataset**
    - Distributions :
      * Taille organoïdes (histogramme log)
      * Nombre cellules/organoïde
      * Qualité segmentation (Dice scores)
      * Distribution temporelle des acquisitions

---

## Récapitulatif par Chapitre

- **Chapitre 1 (Introduction)** : 7 figures essentielles
- **Chapitre 2 (État de l'art)** : 6 figures essentielles
- **Chapitre 3 (Fondements)** : 12 figures essentielles
- **Chapitre 4 (Méthodologie)** : 17 figures essentielles
- **Chapitre 5 (Résultats)** : 30 figures essentielles
- **Chapitre 6 (Conclusion)** : 2 figures essentielles
- **Annexes** : 8 figures complémentaires

**Total : 82 figures**

## Priorités

### Figures Absolument Essentielles (Priority 1 - 25 figures)
1.5, 1.6, 1.7, 4.1, 4.4, 4.5, 4.6, 4.8, 4.11, 4.14, 4.16, 5.1, 5.3, 5.8, 5.10, 5.16, 5.17, 5.20, 5.24, 5.29

### Figures Importantes (Priority 2 - 30 figures)
1.1, 2.4, 3.4, 3.9, 3.10, 4.7, 4.10, 4.12, 4.13, 4.15, 5.4, 5.12, 5.13, 5.15, 5.18, 5.19, 5.21, 5.22, 5.23, 5.26, 5.28

### Figures Complémentaires (Priority 3 - 27 figures)
Toutes les autres figures listées ci-dessus

---

## Recommandations de Style

1. **Cohérence visuelle** : Palette de couleurs uniforme dans toute la thèse
   - Chouxfleurs : Orange/Rouge
   - Cystiques : Bleu/Cyan
   - Compact : Vert
   - Kératinisés : Violet

2. **Qualité** : 
   - Résolution minimale 300 DPI
   - Format vectoriel (SVG/PDF) pour schémas
   - Format raster haute résolution pour images microscopiques

3. **Légendes** :
   - Complètes et auto-suffisantes
   - Définition de tous les acronymes
   - Indication des barres d'échelle pour images microscopiques

4. **Accessibilité** :
   - Colormaps adaptées aux daltoniens (viridis, plasma)
   - Texte lisible (taille ≥ 8pt)
   - Contraste suffisant

5. **Annotations** :
   - Flèches et labels clairs
   - Numérotation des panels (A, B, C...)
   - Statistiques importantes affichées directement

---

## Notes pour la Production

- **Logiciels recommandés** :
  - Matplotlib/Seaborn (Python) pour graphiques scientifiques
  - Plotly pour visualisations 3D interactives
  - BioRender pour schémas biologiques
  - Inkscape/Adobe Illustrator pour assemblage et retouches
  - napari pour visualisations microscopie

- **Formats sources à conserver** :
  - .py (scripts de génération)
  - .svg (schémas vectoriels)
  - .ai (fichiers Illustrator)
  - .tif (images brutes haute résolution)

- **Workflow** :
  1. Génération automatique des graphiques (scripts Python reproductibles)
  2. Export en format intermédiaire haute résolution
  3. Assemblage et annotation dans Illustrator
  4. Export final PDF pour inclusion LaTeX

