# Plan de Soutenance de Thèse

## Informations générales

- **Titre** : Apprentissage profond pour l'analyse des organoïdes : modélisation par graphes des architectures cellulaires 3D
- **Candidat** : Alexandre Martin
- **Date de soutenance** : 17 décembre 2025
- **Discipline** : Informatique
- **Durée estimée de la présentation** : 45 minutes

---

## Structure de la présentation

### 1. Introduction (6-7 min)

#### 1.1 Accroche et contexte (2 min)
- Les organoïdes : mini-organes 3D cultivés *in vitro*
- Applications : médecine personnalisée, criblage de médicaments
- Visualisation 3D d'organoïdes représentatifs (cystique vs choux-fleurs)

#### 1.2 Problématique scientifique (2-3 min)
- **Verrou 1** : Quantification automatique de structures 3D complexes
  - Analyse manuelle non scalable (15-30 min/organoïde)
- **Verrou 2** : Rareté des données annotées
- **Verrou 3** : Robustesse aux variations expérimentales et géométriques

#### 1.3 Questions de recherche et plan (2 min)
- Q1 : Les graphes géométriques sont-ils une représentation efficace ?
- Q2 : Les GNNs équivariants surpassent-ils les approches classiques ?
- Q3 : Le transfer learning depuis données synthétiques est-il efficace ?
- Annonce du plan en 5 parties

---

### 2. Fondements théoriques : Graphes et Graph Neural Networks (12-14 min)

#### 2.1 Limites des approches existantes (2 min)
- **Descripteurs manuels + ML classique**
  - Feature engineering coûteux, perte d'information relationnelle
- **CNN 3D**
  - Empreinte mémoire prohibitive (~2 Go/organoïde)
  - Downsampling destructif, pas d'invariance géométrique native
- Tableau comparatif rapide des approches

#### 2.2 Représentation par graphes : une abstraction naturelle (2 min)
- Les cellules forment un réseau d'interactions spatiales
- L'organisation relationnelle détermine le phénotype
- Abstraction : cellule → nœud, voisinage → arête
- **Compression spectaculaire** : Go → Mo (facteur 1000)
- Schéma : image 3D → nuage de points → graphe

#### 2.3 Introduction aux Graph Neural Networks (3 min)
- **Problème** : comment apprendre sur des données non-euclidiennes ?
- **Paradigme du Message Passing** (schéma animé)
  - Chaque nœud agrège l'information de ses voisins
  - Formule : h_i^{l+1} = UPDATE(h_i^l, AGGREGATE({h_j^l : j ∈ N(i)}))
  - Empilement de L couches → champ réceptif de rayon L
- **Pooling global** : agrégation au niveau du graphe entier
  - Mean/Sum pooling → vecteur de représentation du graphe
  - Classification/régression sur ce vecteur

#### 2.4 Panorama des architectures GNN testées (4 min)
- **GCN** (Graph Convolutional Network) - Baseline
  - Agrégation par moyenne normalisée des voisins
  - Simple mais traite tous les voisins également
- **GAT** (Graph Attention Network) - Notre choix principal
  - Mécanisme d'attention : apprend à pondérer les voisins
  - Multi-head attention : capture différents types de relations
  - Schéma du calcul des poids d'attention
- **DeepSets** - Comparaison avec approche ensembliste
  - Agrégation globale sans structure de voisinage explicite
  - Baseline pour évaluer l'apport de la structure locale
- **EGNN** (Equivariant GNN) - Pour la robustesse géométrique
  - Messages invariants + mise à jour équivariante des coordonnées
  - Garantie théorique d'équivariance E(3)

#### 2.5 Équivariance géométrique : concept clé (3 min)
- **Pourquoi c'est crucial pour les organoïdes ?**
  - Pas d'orientation privilégiée en culture 3D
  - La prédiction ne doit pas dépendre de l'orientation arbitraire
- **Invariance vs Équivariance** (schéma)
  - Invariance : f(T(x)) = f(x) → sortie identique
  - Équivariance : f(T(x)) = T(f(x)) → sortie se transforme de façon cohérente
- **Groupe E(3)** : rotations + translations + réflexions
- **Comment EGNN garantit l'équivariance ?**
  - Utilise uniquement des quantités invariantes (distances, angles)
  - Mise à jour des coordonnées par vecteurs équivariants
- **Avantages pratiques** :
  - Pas besoin d'augmentation de données pour les rotations
  - Meilleure généralisation avec moins de données
  - Robustesse garantie par construction (pas apprise)

---

### 3. Méthodologie : Pipeline de bout en bout (8-10 min)

#### 3.1 Vue d'ensemble du pipeline (2 min)
- Schéma du flux de données complet :
  1. Image 3D confocale (~2 Go) → Prétraitement
  2. Segmentation cellulaire (Faster Cellpose)
  3. Clustering DBSCAN → Construction de graphes K-NN
  4. Classification GNN → Prédiction du phénotype
- Temps total : ~20 min/organoïde (dominé par segmentation)

#### 3.2 Dataset et segmentation optimisée (3 min)
- **Dataset collaboratif** (ANR Morpheus, IPMC Nice, Paris Cité)
  - 2,272 organoïdes extraits → 500 sélectionnés (qualité)
  - Phénotypes : Cystique vs Choux-fleurs
- **Optimisation segmentation** (contribution)
  - Problème : Cellpose standard = 2,500 heures pour 1,000 organoïdes
  - Faster Cellpose : 5× plus rapide, F1=0.96 préservé
  - Knowledge distillation + pruning 30%

#### 3.3 Construction des graphes et génération synthétique (3 min)
- **Graphes géométriques**
  - Features : coordonnées 3D + volume (4D par nœud)
  - Connectivité : K-NN (k=10) + cutoff radial
- **Génération synthétique par processus ponctuels** (contribution)
  - Poisson homogène → cystique | Matérn cluster → choux-fleurs
  - 100,000 organoïdes synthétiques générés
  - Stratégie : pré-entraînement sur synthétiques → fine-tuning sur réels

---

### 4. Résultats expérimentaux (10-12 min)

#### 4.1 Justification des GNN : étude comparative GRETSI 2025 (3 min)
- **Protocole** : données synthétiques, vérité terrain parfaite
- **Résultat 1 - Robustesse au bruit** :
  - Statistiques spatiales (Ripley) : supérieures sous bruit gaussien
  - GNN : plus sensibles mais récupèrent avec plus de données
- **Résultat 2 - Généralisation géométrique** (graphique clé)
  - Test sur ellipsoïdes (entraînement sur sphères uniquement)
  - Statistiques spatiales : 1.0 → 0.65 (chute de 35%)
  - GNN : 0.95 → 0.82 (dégradation gracieuse)
  - **Conclusion** : GNN adaptés aux morphologies variables réelles

#### 4.2 Comparaison des architectures GNN (4 min)
- **Tâche** : régression du degré d'agrégation sur 100,000 synthétiques
- **Tableau de résultats** :

| Architecture | MSE | Gain vs GCN | Points clés |
|--------------|-----|-------------|-------------|
| GCN | 0.198 | — | Baseline, agrégation uniforme |
| DeepSets | 0.145 | +27% | Pas de structure locale |
| EGNN | 0.137 | +31% | Équivariance E(3) garantie |
| **GAT** | **0.118** | **+40%** | Attention = pondération adaptative |

- **Analyse des résultats** :
  - GAT surpasse grâce au mécanisme d'attention
  - DeepSets étonnamment bon → statistiques globales informatives
  - EGNN : trade-off performance/robustesse géométrique
- **Études d'ablation** :
  - Profondeur optimale : 5 couches (over-smoothing au-delà)
  - Importance de l'équivariance : MSE ÷ 2.8 vs coordonnées brutes

#### 4.3 Performances sur données réelles et transfer learning (3 min)
- **Résultat principal** : 84% d'accuracy (GAT pré-entraîné)
- **Impact du transfer learning** :

| Stratégie | Accuracy | Gain |
|-----------|----------|------|
| GAT from scratch | 76% | — |
| GAT pré-entraîné synthétiques | **84%** | **+8%** |

- Matrice de confusion (visualisation)
- Analyse des erreurs : confusions sur phénotypes intermédiaires
- **Efficacité** : >200 organoïdes/minute (inférence)

#### 4.4 Synthèse : pourquoi GAT + transfer learning ? (2 min)
- GAT : meilleure performance grâce à l'attention adaptative
- Transfer learning : réduction de 75% du besoin en données annotées
- Convergence 3× plus rapide
- Courbes d'apprentissage comparatives (graphique)

---

### 5. Conclusion et perspectives (6-8 min)

#### 5.1 Synthèse des contributions (3 min)
1. **Pipeline automatisé de bout en bout**
   - De l'image brute à la prédiction
   - Compression 1000× préservant l'information biologique
2. **Optimisation de la segmentation**
   - Faster Cellpose : 5× plus rapide
   - Méthode géométrique : 15× plus rapide
3. **Représentation par graphes géométriques**
   - Capture explicite de la structure relationnelle
   - Architectures GNN équivariantes (GAT, EGNN)
4. **Génération synthétique par processus ponctuels**
   - 100,000 organoïdes avec labels contrôlés
   - Transfer learning efficace (+8% accuracy)
5. **Validation expérimentale rigoureuse**
   - Étude comparative GRETSI 2025
   - 84% accuracy sur organoïdes réels

#### 5.2 Limitations identifiées (1-2 min)
- Dépendance à la qualité de segmentation
- Généralisation à d'autres types d'organoïdes à valider
- Pas de validation inter-laboratoires
- Absence d'analyse d'interprétabilité approfondie

#### 5.3 Perspectives à court terme (2 min)
- **Extensions méthodologiques**
  - Graphes multi-échelles (cellule → région → organoïde)
  - Architectures Graph Transformers
  - Caractérisation morphologique par alpha-shapes
- **Intégration multi-modale**
  - Transcriptomique spatiale + imagerie
  - Données temporelles (time-lapse)
- **Validation clinique**
  - Études prospectives sur cohortes de patients
  - Intégration dans workflows hospitaliers

#### 5.4 Vision à long terme (1-2 min)
- **Prédiction de réponse thérapeutique**
  - Organoïdes patient-dérivés → test de traitements
  - Médecine de précision personnalisée
- **Modèles génératifs de graphes**
  - Design rationnel d'organoïdes *in silico*
  - Exploration de l'espace morphologique
- **Impact sociétal**
  - Réduction de l'expérimentation animale (principe des 3R)
  - Accélération du développement de médicaments

#### 5.5 Message final (30 sec)
- Les GNN géométriques : outil puissant pour l'analyse d'organoïdes 3D
- Synergie biologie-IA : cercle vertueux d'amélioration
- Code open-source disponible pour la communauté

---

## Annexes pour les questions du jury

### Questions anticipées et éléments de réponse

#### Sur la méthodologie
- **Pourquoi GAT plutôt qu'EGNN ?**
  - GAT : meilleures performances brutes
  - EGNN : trade-off intéressant pour robustesse géométrique
  - Choix dépend de l'application (précision vs robustesse)

- **Validation statistique des données synthétiques ?**
  - Limitation reconnue : pas de validation formelle
  - Mais gains empiriques observés (+8%) suggèrent utilité pratique

#### Sur les résultats
- **Pourquoi seulement 500 organoïdes sélectionnés sur 2,272 ?**
  - Incohérence label-morphologie dans le reste
  - Qualité > quantité pour apprentissage supervisé
  - Reste réservé pour approches non-supervisées futures

- **Généralisation à d'autres organoïdes ?**
  - Principes transférables (graphes, processus ponctuels)
  - Adaptations nécessaires (fine-tuning Cellpose, features spécifiques)

#### Sur les perspectives
- **Validation clinique concrète ?**
  - Prochaine étape : cohorte 100-500 patients
  - Prédiction réponse thérapeutique ex vivo
  - Certification dispositif médical (IVDR/FDA)

---

## Timing recommandé

| Section | Durée | Cumul | Détail |
|---------|-------|-------|--------|
| 1. Introduction | 6-7 min | 7 min | Contexte, verrous, questions |
| 2. **Fondements GNN** | **12-14 min** | 20 min | **Section clé : théorie des GNN** |
| 3. Méthodologie | 8-10 min | 29 min | Pipeline, données, synthétiques |
| 4. Résultats | 10-12 min | 40 min | Comparaisons, performances |
| 5. Conclusion | 6-8 min | 47 min | Contributions, perspectives |
| **Total présentation** | **45-50 min** | | |
| Questions du jury | 30-45 min | | |
| **Total soutenance** | **~1h30** | | |

> ⚠️ **Note** : La section 2 (Fondements GNN) est volontairement longue car elle présente la contribution théorique centrale de la thèse.

---

## Conseils de présentation

### Slides clés à soigner

#### Section GNN (priorité haute - cœur scientifique)
1. **Message Passing animé** : schéma montrant l'agrégation itérative
2. **Comparaison des architectures** : GCN vs GAT vs EGNN en un coup d'œil
3. **Équivariance E(3)** : schéma rotation → même prédiction
4. **Mécanisme d'attention GAT** : visualisation des poids d'attention sur un graphe

#### Autres slides essentielles
5. **Slide d'accroche** : visualisation 3D impressionnante d'organoïdes
6. **Schéma du pipeline** : vue d'ensemble claire et mémorable
7. **Tableau comparatif des architectures** : MSE et gains en %
8. **Généralisation géométrique** : graphique accuracy vs ratio d'aspect
9. **Slide de synthèse** : les 5 contributions principales

### Points à mettre en valeur
- **Contribution théorique** : premier framework GNN géométrique pour organoïdes 3D
- **Maîtrise des GNN** : comparaison rigoureuse de 4 architectures
- **Équivariance** : garantie formelle, pas juste augmentation de données
- Résultats quantitatifs solides (84% accuracy, +8% gain, MSE ÷2.8)
- Collaboration interdisciplinaire (informatique + biologie)

### Démonstrations possibles
- **Animation message passing** : propagation de l'information dans le graphe
- Visualisation 3D interactive des graphes avec poids d'attention
- Comparaison avant/après : coordonnées brutes vs EGNN équivariant
- Vidéo du pipeline en action

### Anticipation des questions sur les GNN
- Préparer des slides de backup sur :
  - Formules mathématiques détaillées (attention, équivariance)
  - Over-smoothing et solutions
  - Comparaison avec Graph Transformers
  - Complexité computationnelle des architectures

