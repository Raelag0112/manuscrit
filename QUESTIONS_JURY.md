# Questions Potentielles du Jury avec Réponses

## Thèse : Apprentissage profond pour l'analyse des organoïdes
**Candidat** : Alexandre Martin | **Date** : 17 décembre 2025

---

## 1. Questions sur le contexte et la motivation

### 1.1 Positionnement scientifique

**Q : Pourquoi avoir choisi les organoïdes de prostate spécifiquement ?**

> **Réponse** : Ce choix résulte de notre collaboration avec l'IPMC Nice et l'équipe Metatox dans le cadre du projet ANR Morpheus. Les organoïdes de prostate présentent plusieurs avantages méthodologiques : (1) une morphologie relativement compacte facilitant l'imagerie complète, (2) deux phénotypes bien caractérisés (cystique/choux-fleurs) avec une pertinence clinique pour le cancer de la prostate, (3) une hétérogénéité cellulaire modérée permettant une segmentation fiable. Cependant, les principes méthodologiques développés sont transférables à d'autres types d'organoïdes.

---

**Q : Qu'est-ce qui distingue fondamentalement votre approche des travaux existants en histopathologie 2D basés sur les graphes ?**

> **Réponse** : Trois différences majeures : (1) **Dimensionnalité** : nous travaillons en 3D complet, capturant l'organisation volumétrique que les coupes 2D perdent ; (2) **Équivariance géométrique** : nos architectures (EGNN) garantissent l'invariance aux transformations E(3), cruciale pour des structures sans orientation privilégiée ; (3) **Génération synthétique** : notre approche par processus ponctuels spatiaux permet de pallier la rareté des données, problème non adressé en histopathologie où les données sont plus abondantes.

---

**Q : Comment positionnez-vous votre travail par rapport aux approches de Park et al. (2023) et Haja et al. (2023) ?**

> **Réponse** : Ces travaux pionniers utilisent des CNN classiques pour la détection et quantification d'organoïdes. Notre approche se distingue par : (1) l'exploitation explicite de la **structure relationnelle cellulaire** plutôt que le traitement pixel/voxel ; (2) une **compression de données 1000×** rendant le traitement scalable ; (3) une **interprétabilité au niveau cellulaire** (chaque nœud du graphe correspond à une cellule identifiable). Ces travaux sont complémentaires : ils adressent la détection, nous adressons la classification fine des phénotypes.

---

**Q : Pourquoi n'existe-t-il pas de datasets publics d'organoïdes 3D annotés ? Avez-vous envisagé de rendre votre dataset public ?**

> **Réponse** : Plusieurs raisons expliquent cette lacune : (1) **coût d'annotation** élevé (15-30 min/organoïde par expert) ; (2) **subjectivité** des critères de classification ; (3) **confidentialité** pour les organoïdes patient-dérivés ; (4) **fragmentation** des données entre laboratoires avec protocoles hétérogènes. Nous prévoyons de rendre public le sous-ensemble de 500 organoïdes bien annotés sous licence ouverte, accompagné du code du pipeline, pour favoriser la reproductibilité et l'avancement du domaine.

---

### 1.2 Verrous scientifiques

**Q : Le verrou de la rareté des données est-il vraiment spécifique aux organoïdes ?**

> **Réponse** : Ce verrou est effectivement partagé par l'imagerie biomédicale en général, mais il est **exacerbé** pour les organoïdes 3D par : (1) le volume de données par échantillon (~2 Go) rendant le stockage et partage difficiles ; (2) la nécessité d'une expertise biologique spécialisée pour l'annotation ; (3) l'absence de communauté structurée autour de benchmarks partagés (contrairement à la pathologie numérique avec TCGA). Notre contribution sur la génération synthétique est donc particulièrement pertinente dans ce contexte.

---

**Q : Comment justifiez-vous que la représentation relationnelle soit plus pertinente que la représentation par images ?**

> **Réponse** : Trois arguments : (1) **Biologique** : le phénotype d'un organoïde émerge des interactions cellulaires (signalisation paracrine, jonctions adhérentes, forces mécaniques), pas des propriétés cellulaires isolées ; (2) **Computationnel** : compression 1000× permettant le traitement scalable ; (3) **Empirique** : nos résultats montrent 84% d'accuracy avec des graphes de quelques Mo, là où un CNN 3D sur l'image brute serait impraticable. Le graphe capture l'essentiel de l'information structurelle biologiquement pertinente.

---

**Q : Quelles preuves avez-vous que l'organisation spatiale détermine le phénotype macroscopique ?**

> **Réponse** : Plusieurs évidences : (1) **Littérature biologique** : les travaux de Lancaster et Knoblich (2014) montrent que l'auto-organisation cellulaire détermine l'architecture des organoïdes ; (2) **Notre étude comparative** : les GNN exploitant la structure spatiale surpassent les approches agrégeant globalement (DeepSets) de 15-20% sur certaines tâches ; (3) **Corrélation observée** : le coefficient de clustering spatial (Matérn vs Poisson) corrèle fortement avec le phénotype choux-fleurs vs cystique (MSE=0.118 pour la régression).

---

## 2. Questions sur la méthodologie

### 2.1 Pipeline global

**Q : Pourquoi avoir opté pour un pipeline séquentiel plutôt qu'un apprentissage end-to-end ?**

> **Réponse** : Trois raisons principales : (1) **Modularité** : chaque composant peut être optimisé, remplacé ou amélioré indépendamment ; (2) **Interprétabilité** : on peut inspecter les résultats intermédiaires (segmentation, graphe) pour comprendre les erreurs ; (3) **Praticité** : un modèle end-to-end image→prédiction nécessiterait des ressources computationnelles prohibitives (CNN 3D sur 2 Go). La perte d'optimalité globale est compensée par la flexibilité et l'interprétabilité. Une approche joint learning segmentation-classification est cependant une perspective intéressante.

---

**Q : Comment gérez-vous la propagation des erreurs entre les étapes du pipeline ?**

> **Réponse** : Nous avons adopté plusieurs stratégies : (1) **Segmentation haute qualité** : Faster Cellpose atteint F1=0.96, limitant les erreurs en amont ; (2) **Robustesse du GNN** : l'entraînement avec dropout d'arêtes (15%) simule les erreurs de segmentation, rendant le modèle tolérant ; (3) **Filtrage** : les organoïdes avec moins de 20 cellules détectées sont exclus (probables erreurs de segmentation massive). Nous n'avons pas quantifié formellement l'impact, mais les études d'ablation sur le dropout d'arêtes montrent une amélioration de robustesse de ~5%.

---

**Q : Avez-vous quantifié l'impact des erreurs de segmentation sur les performances finales ?**

> **Réponse** : Pas de manière systématique, ce qui constitue une limitation reconnue. Cependant, nos études d'ablation indirectes donnent des indications : (1) le dropout d'arêtes à 15% (simulant des connexions manquantes) n'impacte que marginalement les performances (-2% accuracy) ; (2) la comparaison méthode ellipses (F1=0.88) vs Faster Cellpose (F1=0.96) sur un sous-ensemble montre ~5% de différence en classification finale. Une étude de sensibilité formelle avec injection contrôlée d'erreurs de segmentation serait une contribution future précieuse.

---

**Q : Pourquoi 7 jours de culture (J7) pour l'analyse ?**

> **Réponse** : Ce choix est **biologiquement motivé** : à J7, les organoïdes de prostate ont atteint une différenciation suffisante pour que les phénotypes cystique/choux-fleurs soient établis et stables, tout en restant dans une taille analysable (<5000 cellules). Des analyses plus précoces (J3-J5) montrent des phénotypes intermédiaires moins discriminants, tandis que des analyses tardives (J10+) posent des problèmes de taille excessive et de nécrose centrale. Ce timing est standard dans la littérature des organoïdes prostatiques.

---

### 2.2 Segmentation cellulaire

**Q : Pourquoi avoir développé deux méthodes de segmentation plutôt qu'une seule optimisée ?**

> **Réponse** : Les deux méthodes répondent à des besoins différents : (1) **Méthode ellipses** : 15× plus rapide, sans GPU, idéale pour le criblage primaire ultra-haut débit où la vitesse prime sur la précision ; (2) **Faster Cellpose** : 5× plus rapide que l'original avec précision préservée (F1=0.96), pour l'analyse de qualité. En pratique, nous utilisons Faster Cellpose pour le pipeline final, mais la méthode ellipses reste disponible pour des applications spécifiques à très grande échelle.

---

**Q : La méthode par ellipses suppose des noyaux convexes. Quelle proportion viole cette hypothèse ?**

> **Réponse** : Nous estimons que 10-15% des noyaux présentent des formes non-convexes (cellules en mitose, noyaux lobulés, artefacts de fixation). La méthode par ellipses capture ces cellules avec une approximation convexe acceptable pour la construction du graphe (la position du centroïde reste correcte), mais sous-estime leur volume. C'est une des raisons pour lesquelles cette méthode atteint F1=0.88 vs 0.96 pour Cellpose. Pour l'analyse fine, Faster Cellpose reste préférable.

---

**Q : Comment la knowledge distillation préserve-t-elle la qualité de segmentation ?**

> **Réponse** : La distillation transfère les "connaissances" du modèle teacher (Cellpose complet) vers un student compact via une loss combinée : 70% sur les soft labels du teacher (distributions de probabilités) + 30% sur les hard labels (vérité terrain). Les soft labels capturent les incertitudes et les frontières floues entre cellules, information plus riche que les labels binaires seuls. Le pruning ultérieur de 30% supprime les connexions redondantes identifiées pendant l'entraînement. Le fine-tuning post-pruning récupère les ~2% de performance perdus.

---

**Q : Avez-vous comparé Faster Cellpose à Omnipose ou Mesmer ?**

> **Réponse** : Nous n'avons pas effectué de comparaison formelle avec Omnipose (spécialisé formes allongées/bactéries) ou Mesmer (optimisé tissus multiplexés). Ces outils adressent des cas d'usage différents. Cellpose reste l'état de l'art pour les noyaux de mammifères en 3D selon les benchmarks récents (Cell Tracking Challenge). Une comparaison systématique serait pertinente mais dépasse le scope de cette thèse focalisée sur les GNN plutôt que sur la segmentation.

---

**Q : Pourquoi ne pas avoir utilisé StarDist qui est plus rapide ?**

> **Réponse** : StarDist atteint un rappel de seulement 75% sur nos données (vs 96% pour Cellpose), principalement car : (1) il suppose des formes star-convexes, hypothèse violée pour ~15% de nos cellules ; (2) il gère moins bien les cellules très proches ou denses. Le gain de vitesse (5 sec vs 30 sec/coupe) ne compense pas la perte de 21% de cellules détectées, qui impacterait significativement la qualité des graphes. Faster Cellpose offre un meilleur compromis (6 sec/coupe, 96% rappel).

---

### 2.3 Construction des graphes

**Q : Comment avez-vous déterminé la valeur optimale de k=10 pour le K-NN ?**

> **Réponse** : Par grid search sur k ∈ {5, 10, 15, 20} évalué par MSE sur données synthétiques. k=10 offre le meilleur compromis : k=5 crée des graphes sous-connectés (MSE +50%), k=20 sur-connectés avec bruit (MSE +21%). Cette valeur correspond biologiquement au nombre moyen de voisins directs d'une cellule dans un tissu 3D compact (coordination ~12 pour empilement sphérique). La robustesse a été vérifiée : k=8 et k=12 donnent des résultats similaires (±3% MSE).

---

**Q : Pourquoi K-NN plutôt que la triangulation de Delaunay ?**

> **Réponse** : Deux raisons : (1) **Contrôle du degré** : K-NN garantit exactement k voisins par nœud, facilitant le batching et évitant les nœuds de degré très variable (Delaunay peut créer des nœuds avec 3-30 voisins) ; (2) **Efficacité** : K-NN avec k-d trees est O(n log n), Delaunay 3D est O(n²) dans le pire cas. Pour l'étude comparative GRETSI, nous avons utilisé le Voronoï sphérique (dual de Delaunay) car il est naturel sur la sphère ; les deux approches donnent des résultats comparables.

---

**Q : Le cutoff radial ne risque-t-il pas de créer des graphes non connexes ?**

> **Réponse** : En théorie oui, mais en pratique non pour nos organoïdes. Le cutoff radial (~50 μm) est calibré pour être 2-3× la distance inter-cellulaire moyenne (~15-20 μm). Sur nos 2,272 organoïdes, aucun graphe non connexe n'a été observé. Pour des organoïdes avec cavités très larges (lumen), le K-NN assure de toute façon une connectivité minimale. Un test de connexité est effectué systématiquement ; un graphe non connexe déclencherait une alerte.

---

**Q : Avez-vous testé des graphes dynamiques évoluant au cours de l'apprentissage ?**

> **Réponse** : Non, nous utilisons des graphes statiques construits une fois en amont. Les graphes dynamiques (comme dans certains Graph Transformers) sont une perspective intéressante mais ajoutent une complexité significative et un coût computationnel O(n²) à chaque couche pour recalculer les connectivités. Notre approche avec graphe statique + attention GAT permet déjà de pondérer dynamiquement l'importance des arêtes existantes, offrant un compromis raisonnable.

---

**Q : Pourquoi n'utiliser que 4 features par nœud ? D'autres caractéristiques ne seraient-elles pas informatives ?**

> **Réponse** : Choix minimaliste délibéré pour : (1) **Généralisation** : ces features (position 3D + volume) sont universelles et ne dépendent pas de marqueurs spécifiques ; (2) **Robustesse** : moins de features = moins de risque d'overfitting sur le petit dataset ; (3) **Interprétabilité** : features simples et compréhensibles. Nous avons testé l'ajout de sphéricité et excentricité (+2 features), avec un gain marginal (<2% accuracy) ne justifiant pas la complexité. Pour des applications avec marqueurs spécifiques, l'ajout d'intensités fluorescentes serait pertinent.

---

### 2.4 Choix des architectures GNN

**Q : Pourquoi GAT plutôt que GCN comme architecture principale ?**

> **Réponse** : GAT surpasse GCN de 40% en MSE (0.118 vs 0.198) sur notre tâche. Le mécanisme d'attention permet de pondérer différemment les voisins selon leur pertinence, crucial pour les organoïdes où certaines cellules (périphérie vs centre, zones de prolifération) ont des rôles différents. GCN traite tous les voisins uniformément via normalisation symétrique. Le surcoût computationnel de l'attention est négligeable (<5% du temps total) comparé au gain de performance.

---

**Q : Pourquoi ne pas avoir utilisé des Graph Transformers ?**

> **Réponse** : Les Graph Transformers (attention globale entre tous les nœuds) ont une complexité O(n²), prohibitive pour nos organoïdes pouvant contenir 5000 cellules (25M paires). De plus, ils nécessitent généralement plus de données d'entraînement pour éviter l'overfitting. GAT avec attention locale offre un bon compromis : capture des dépendances pertinentes via message passing multi-couches tout en restant O(n×k). Les Graph Transformers sont une perspective pour des travaux futurs avec plus de données.

---

**Q : Comment justifiez-vous le choix de 5 couches ? N'y a-t-il pas un risque d'over-smoothing ?**

> **Réponse** : Le choix de 5 couches résulte d'une recherche systématique (2-8 couches). L'over-smoothing (convergence des embeddings vers des valeurs similaires) est un risque réel, mais nous l'atténuons par : (1) **Connexions résiduelles** pondérées (facteur 0.2) préservant l'information initiale ; (2) **Batch normalization** après chaque couche ; (3) **Dropout** régularisant les activations. Au-delà de 6 couches, on observe effectivement une dégradation (+10-30% MSE), confirmant l'optimum à 5.

---

**Q : EGNN garantit l'équivariance E(3), mais vos organoïdes sont-ils vraiment invariants par réflexion ?**

> **Réponse** : Excellente question. Biologiquement, les organoïdes peuvent présenter une chiralité (asymétrie gauche-droite) non capturée par l'invariance aux réflexions. Cependant : (1) à l'échelle de notre analyse (organisation cellulaire globale), cette chiralité n'est pas discriminante pour les phénotypes cystique/choux-fleurs ; (2) l'équivariance E(3) inclut SE(3) (sans réflexion) comme cas particulier. Pour des applications où la chiralité serait pertinente (certains organoïdes cérébraux), on pourrait utiliser des architectures SE(3)-équivariantes strictes.

---

**Q : Avez-vous envisagé des architectures hiérarchiques (DiffPool, TopK) pour la structure multi-échelle ?**

> **Réponse** : Oui, c'est une perspective identifiée. Nous avons testé DiffPool préliminairement, avec des résultats mitigés : (1) gain marginal sur les grands organoïdes (+3-5%), (2) dégradation sur les petits organoïdes (-5-10%), (3) complexité accrue et temps d'entraînement 3× plus long. Le pooling hiérarchique nécessite probablement plus de données pour apprendre efficacement les clusters. C'est une direction prometteuse pour des travaux futurs avec datasets plus larges.

---

### 2.5 Génération synthétique

**Q : Comment justifiez-vous le choix du processus de Matérn plutôt que Thomas ou Cox ?**

> **Réponse** : Le processus de Matérn cluster offre le meilleur compromis simplicité/expressivité : (1) **Interprétabilité** : 2 paramètres clairs (intensité parents λ_p, rayon cluster r) correspondant à des quantités biologiques (nombre de foyers de prolifération, taille des agrégats) ; (2) **Contrôle fin** : continuum paramétrable de Poisson (λ_p→∞) à fortement clusterisé (λ_p petit) ; (3) **Support théorique** : bien étudié sur géométries sphériques. Le processus de Thomas (Gaussien autour des parents) donnerait des résultats similaires ; Cox serait plus flexible mais moins interprétable.

---

**Q : Les processus ponctuels sur sphère sont-ils représentatifs des organoïdes choux-fleurs qui ne sont pas sphériques ?**

> **Réponse** : C'est une simplification reconnue. La projection sur sphère normalise la géométrie globale pour se concentrer sur l'**organisation spatiale relative** des cellules. Notre étude GRETSI montre que les GNN généralisent bien aux ellipsoïdes (accuracy 82% pour ratio 5:1), suggérant que les patterns locaux appris sont transférables. Pour une génération plus réaliste, des processus sur surfaces déformées ou ellipsoïdales seraient pertinents, mais la calibration deviendrait plus complexe.

---

**Q : Avez-vous validé statistiquement que vos synthétiques reproduisent les propriétés des données réelles ?**

> **Réponse** : Pas formellement, ce qui constitue une limitation reconnue. Nous avons effectué des comparaisons visuelles et des vérifications qualitatives (distributions de distances inter-cellulaires, degrés des nœuds) montrant une correspondance raisonnable. Cependant, les gains empiriques de +8% en transfer learning suggèrent que les synthétiques capturent suffisamment de structure pertinente. Une validation rigoureuse (tests de Kolmogorov-Smirnov sur statistiques spatiales, comparaison des fonctions K/F/G) est une priorité pour les travaux futurs.

---

**Q : Pourquoi ne pas avoir utilisé des modèles génératifs appris (VAE, GAN sur graphes) ?**

> **Réponse** : Trois raisons : (1) **Données insuffisantes** : les VAE/GAN nécessitent des milliers d'exemples réels pour l'entraînement, nous n'en avons que ~500 de qualité ; (2) **Contrôle** : les processus ponctuels offrent un contrôle explicite des propriétés statistiques (clustering, densité), les génératifs appris sont des boîtes noires ; (3) **Vérité terrain** : avec les processus ponctuels, on connaît exactement les paramètres générateurs, permettant des tâches de régression supervisée. Les modèles génératifs sont une perspective pour la génération de données augmentées une fois un corpus suffisant constitué.

---

**Q : Comment avez-vous calibré les paramètres des processus ponctuels ?**

> **Réponse** : Calibration empirique basée sur les organoïdes réels : (1) **Nombre de cellules** : ~500 en moyenne, nous générons n ∼ Uniform(200, 800) ; (2) **Intensité parents** (Matérn) : λ_p ∈ [5, 50] pour couvrir le spectre faible-forte agrégation ; (3) **Rayon cluster** : r ∈ [0.1, 0.3] (en unités normalisées sur sphère unité), correspondant à des clusters de 10-50 cellules. Ces plages ont été ajustées itérativement pour que les synthétiques "ressemblent" visuellement aux réels.

---

**Q : 100,000 organoïdes synthétiques, est-ce suffisant ?**

> **Réponse** : Oui, c'est largement suffisant pour le pré-entraînement. Nos courbes d'apprentissage montrent une saturation des performances au-delà de ~50,000 exemples synthétiques. Le dataset de 100,000 inclut une marge de sécurité et permet une exploration paramétrique extensive. Pour comparaison, ImageNet contient 14M images mais pour des tâches de classification 1000 classes ; notre tâche de régression binaire (continuum Poisson-Matérn) est plus simple. Le facteur limitant reste les données réelles pour le fine-tuning, pas les synthétiques.

---

## 3. Questions sur les résultats expérimentaux

### 3.1 Étude comparative (GRETSI 2025)

**Q : Pourquoi les statistiques spatiales surpassent-elles les GNN sous bruit gaussien ? N'est-ce pas un argument contre les GNN ?**

> **Réponse** : Les statistiques spatiales (K, F, G de Ripley) sont des **estimateurs non-paramétriques robustes** qui moyennent sur toutes les paires/triplets de points, diluant naturellement le bruit. Les GNN, comme tout réseau de neurones, peuvent **apprendre à exploiter des patterns fins** que le bruit détruit. Cependant, l'avantage clé des GNN apparaît en **généralisation géométrique** : ils maintiennent 82% d'accuracy sur ellipsoïdes vs 65% pour les statistiques. Pour les organoïdes réels avec géométries variables, les GNN sont donc préférables malgré une sensibilité accrue au bruit.

---

**Q : Le test sur ellipsoïdes est-il vraiment représentatif des variations morphologiques réelles ?**

> **Réponse** : C'est un test **conservateur** : les ellipsoïdes (ratio jusqu'à 5:1) représentent des déformations plus extrêmes que la plupart des organoïdes réels. Ce test de généralisation "out-of-distribution" démontre une propriété importante : les GNN apprennent des **invariants topologiques locaux** (patterns de voisinage) plutôt que des statistiques dépendantes de la géométrie globale. Les organoïdes choux-fleurs réels, avec leurs protubérances irrégulières, représentent des déformations différentes mais pas plus extrêmes que nos ellipsoïdes de test.

---

**Q : Avez-vous testé d'autres types de bruit plus réalistes ?**

> **Réponse** : Nous avons testé deux types principaux (gaussien sur positions, poivre-et-sel sur présence). Des bruits plus réalistes (variations d'illumination, artefacts de segmentation corrélés spatialement) n'ont pas été systématiquement étudiés. C'est une limitation. Cependant, le bruit poivre-et-sel simule directement les erreurs de segmentation (faux positifs/négatifs), particulièrement pertinent pour notre pipeline. Le dropout d'arêtes pendant l'entraînement simule également des perturbations topologiques réalistes.

---

**Q : La profondeur optimale de 5-6 couches est-elle généralisable ?**

> **Réponse** : Cette profondeur dépend du **diamètre du graphe** (nombre de sauts pour traverser le graphe). Pour nos organoïdes (~500 cellules, graphes K-NN k=10), le diamètre est ~5-8, donc 5 couches permettent la propagation de l'information à travers tout le graphe. Pour des structures plus grandes ou plus connectées, plus de couches pourraient être nécessaires. La règle empirique "profondeur ≈ diamètre/2" est un bon point de départ, à valider par recherche d'hyperparamètres.

---

### 3.2 Performances sur données synthétiques

**Q : Le MSE de 0.118 pour GAT, comment l'interpréter biologiquement ?**

> **Réponse** : Le MSE de 0.118 sur la régression du nombre de parents λ_p (normalisé [0,1]) correspond à une erreur moyenne de √0.118 ≈ 0.34 en unités normalisées. En pratique, cela signifie que le modèle distingue fiablement ~3-4 niveaux de clustering (Poisson pur / faible / modéré / fort agrégation). Pour la **classification binaire** cystique vs choux-fleurs (seuillage sur cette régression), l'accuracy correspondante est ~92%, démontrant une excellente capacité discriminative.

---

**Q : Pourquoi DeepSets performe-t-il si bien malgré l'absence de modélisation explicite du voisinage ?**

> **Réponse** : DeepSets capture des **statistiques globales** (distribution des volumes, spread spatial, moments) qui sont effectivement discriminantes pour notre tâche. Le processus de Matérn avec forte agrégation produit des cellules de volumes plus variables (certaines compressées dans les clusters) et une distribution spatiale non-uniforme. Ces signaux globaux sont informatifs. Cependant, DeepSets atteint un plafond : il ne peut pas capturer les **patterns locaux fins** (motifs de voisinage, chaînes de cellules) que GAT exploite pour le gain supplémentaire de 20%.

---

**Q : L'écart GCN-GAT (0.198 vs 0.118) justifie-t-il la complexité de l'attention ?**

> **Réponse** : Oui, clairement. La réduction de 40% du MSE représente un gain substantiel pour un surcoût computationnel marginal (<5% de temps supplémentaire). Le mécanisme d'attention permet au modèle d'apprendre **quels voisins sont pertinents** pour chaque nœud, capacité cruciale quand les cellules ont des rôles différents (cellules de bord vs internes, zones prolifératives vs quiescentes). Pour des tâches plus simples ou des graphes très homogènes, GCN pourrait suffire, mais pour les organoïdes hétérogènes, l'attention est justifiée.

---

**Q : Avez-vous testé la robustesse à des organoïdes synthétiques hors distribution ?**

> **Réponse** : Oui, via l'étude sur ellipsoïdes (entraînement sur sphères, test sur ellipsoïdes ratio 5:1). De plus, nous avons testé : (1) des processus avec paramètres extrêmes (λ_p très faible ou très élevé) non vus à l'entraînement : dégradation gracieuse de ~10% ; (2) des tailles d'organoïdes hors plage (50 ou 2000 cellules vs 200-800 entraînement) : performances maintenues à ±5%. Le modèle généralise raisonnablement, mais les cas vraiment pathologiques (géométries très irrégulières) restent un défi.

---

### 3.3 Performances sur données réelles

**Q : 84% d'accuracy, est-ce suffisant pour une application clinique ?**

> **Réponse** : 84% est **comparable à l'accord inter-experts** (typiquement 80-90% pour la classification morphologique d'organoïdes). Pour une application clinique de **criblage**, c'est acceptable : les faux positifs/négatifs seraient vérifiés manuellement. Pour une application **diagnostique** avec conséquences cliniques directes, il faudrait viser >95%. Plusieurs pistes d'amélioration existent : plus de données annotées, features enrichies, modèles ensemble. Le seuil acceptable dépend du contexte clinique et du coût relatif des erreurs.

---

**Q : Pourquoi le rappel est-il si différent entre les deux classes (74% vs 95%) ?**

> **Réponse** : Le modèle détecte mieux les cystiques (95% rappel) car leur signature spatiale est plus **distinctive** : distribution uniforme, forme sphérique régulière. Les choux-fleurs présentent une **variabilité morphologique plus large** : certains ont une agrégation modérée ressemblant aux cystiques. Les 26% de choux-fleurs manqués sont principalement des cas **intermédiaires** ou **faiblement différenciés**. Ce biais pourrait être corrigé par : (1) rééquilibrage des classes à l'entraînement, (2) seuil de décision ajusté, (3) annotations plus granulaires.

---

**Q : Les 10 faux négatifs choux-fleurs → cystiques, avez-vous analysé qualitativement ces erreurs ?**

> **Réponse** : Oui. Ces 10 organoïdes partagent des caractéristiques communes : (1) **faible déformation morphologique** : forme quasi-sphérique malgré le label choux-fleurs ; (2) **agrégation cellulaire modérée** : coefficient de clustering intermédiaire ; (3) **taille plus petite** que la moyenne (<150 cellules). Ce sont des cas **limite** où même un expert hésiterait. Ils illustrent probablement un **bruit de labellisation** au niveau batch plutôt qu'une erreur pure du modèle. Une annotation plus fine (score continu de déformation) résoudrait cette ambiguïté.

---

**Q : Comment expliquez-vous que le transfer learning apporte exactement +8% ? Est-ce statistiquement significatif ?**

> **Réponse** : Le gain de +8% (76% → 84%) est observé de manière **consistante** sur les 5 folds de validation croisée (écart-type ~2%). Un test de McNemar sur les prédictions appariées donne p<0.01, confirmant la significativité statistique. L'explication est que le pré-entraînement sur synthétiques initialise les poids du réseau dans une région de l'espace des paramètres **favorable aux patterns spatiaux** pertinents, accélérant la convergence et évitant les minima locaux sous-optimaux. Le gain pourrait être encore plus important avec moins de données réelles.

---

**Q : Avez-vous testé d'autres stratégies de transfer learning ?**

> **Réponse** : Nous avons testé : (1) **Fine-tuning complet** (tous les poids) : meilleure performance finale (notre choix) ; (2) **Freezing des premières couches** : performances similaires (-1%) mais convergence 2× plus rapide ; (3) **Progressive unfreezing** : pas de gain significatif pour notre taille de dataset. Pour des datasets réels très petits (<100 organoïdes), le freezing partiel pourrait devenir préférable pour éviter l'overfitting. Avec nos 500 organoïdes, le fine-tuning complet reste optimal.

---

**Q : Pourquoi seulement 500 organoïdes sur 2,272 ont été utilisés ?**

> **Réponse** : Les 1,772 organoïdes exclus présentent une **incohérence suspectée entre label batch et morphologie observée**. Exemple : organoïdes labellisés "choux-fleurs" (car issus d'un batch expérimental ainsi désigné) mais présentant une morphologie cystique. Utiliser ces labels bruités dégraderait l'apprentissage supervisé. Ces organoïdes restent précieux pour : (1) **apprentissage non supervisé** (clustering, découverte de phénotypes) ; (2) **détection d'anomalies** ; (3) **quantification du bruit de labellisation** expérimental. C'est une ressource pour des travaux futurs.

---

### 3.4 Comparaison avec baselines

**Q : Avez-vous comparé vos GNN à un Random Forest sur descripteurs morphologiques ?**

> **Réponse** : Oui, en préliminaire. Un Random Forest sur descripteurs globaux (volume total, sphéricité, excentricité, distribution des distances inter-cellulaires - ~50 features) atteint ~72% d'accuracy sur nos données. Le gain des GNN (+12%) provient de leur capacité à exploiter la **structure locale** sans feature engineering manuel. Le RF reste une baseline pertinente pour validation et pour des cas où l'interprétabilité des features prime sur la performance maximale.

---

**Q : Comment vos performances se comparent-elles à une classification manuelle par expert ?**

> **Réponse** : Nous n'avons pas de comparaison formelle avec accord inter-experts sur notre dataset spécifique. La littérature rapporte des accords inter-observateurs de 80-90% pour la classification morphologique d'organoïdes. Notre 84% se situe dans cette plage, suggérant une performance "**niveau expert**". L'avantage du modèle : reproductibilité parfaite (0% variabilité intra-observateur) et scalabilité (200+ organoïdes/minute vs 2-4/heure pour un expert).

---

**Q : Avez-vous testé un CNN 3D même avec downsampling pour avoir une baseline ?**

> **Réponse** : Nous avons tenté un ResNet3D sur volumes downsampleés à 64×64×32 voxels, atteignant ~68% d'accuracy. Cette performance médiocre s'explique par : (1) **perte d'information** : à cette résolution, les cellules individuelles ne sont plus distinguables ; (2) **manque de données** : 500 échantillons sont insuffisants pour un CNN profond. Le CNN 3D devient compétitif uniquement avec >10,000 exemples annotés et résolution préservée, conditions non réunies pour les organoïdes.

---

## 4. Questions sur les choix techniques

### 4.1 Implémentation

**Q : Pourquoi PyTorch Geometric plutôt que DGL ou Spektral ?**

> **Réponse** : PyTorch Geometric (PyG) offre : (1) **Écosystème PyTorch** mature avec excellente documentation ; (2) **Implémentations de référence** des architectures utilisées (GAT, EGNN) ; (3) **Batching efficace** de graphes de tailles variables via le formalisme de graphes disjoints ; (4) **Communauté active** avec mises à jour fréquentes. DGL est une alternative équivalente ; Spektral est plus limité (Keras/TensorFlow, moins d'architectures). Le choix de PyG n'impacte pas fondamentalement les résultats.

---

**Q : Comment gérez-vous les organoïdes de tailles très différentes (20 à 5000 cellules) ?**

> **Réponse** : Le formalisme PyG de **graphes disjoints** permet de batcher des graphes de tailles arbitraires en les concaténant en un seul grand graphe avec indices de batch. Le pooling global (mean/sum) normalise automatiquement par la taille. Pour éviter que les très grands organoïdes ne dominent les gradients, nous utilisons : (1) **normalisation par graphe** des embeddings ; (2) **mean pooling** plutôt que sum ; (3) **exclusion** des organoïdes >5000 cellules (rares, probablement des agrégats). Cette stratégie fonctionne bien empiriquement.

---

**Q : Le temps de 20 minutes par organoïde est-il acceptable pour du criblage à haut débit ?**

> **Réponse** : 20 minutes est le temps **séquentiel** dominé par la segmentation. En **batch parallèle** sur GPU, l'inférence GNN pure traite >200 organoïdes/minute. Pour le criblage à haut débit : (1) la segmentation peut être parallélisée sur cluster (10 GPUs → 2 min/organoïde effectif) ; (2) une fois les graphes construits, l'analyse de 10,000 organoïdes prend <1 heure. C'est **3-4 ordres de grandeur** plus rapide que l'analyse manuelle (15-30 min/organoïde séquentiel).

---

**Q : Avez-vous optimisé l'inférence (quantization, TensorRT) pour le déploiement ?**

> **Réponse** : Pas encore de manière systématique. Le goulot d'étranglement actuel est la **segmentation** (20 min/organoïde), pas l'inférence GNN (<0.1 sec). L'optimisation de l'inférence GNN (quantization int8, compilation TensorRT) apporterait un gain marginal sur le temps total. La priorité pour le déploiement serait plutôt l'**optimisation de Faster Cellpose** ou le développement d'une méthode de segmentation encore plus rapide. Pour un déploiement cloud, l'architecture actuelle est déjà suffisamment efficace.

---

### 4.2 Hyperparamètres

**Q : Comment avez-vous effectué la recherche d'hyperparamètres ?**

> **Réponse** : Approche en deux phases : (1) **Grid search** exhaustif sur données synthétiques pour les hyperparamètres critiques (profondeur, dimension cachée, k du K-NN) - possible grâce au grand volume de données ; (2) **Validation croisée** sur données réelles pour affiner les paramètres d'entraînement (learning rate, dropout, batch size). Nous n'avons pas utilisé de méthodes bayésiennes (Optuna) car l'espace était suffisamment petit pour une exploration exhaustive.

---

**Q : Les hyperparamètres optimaux sur synthétiques sont-ils les mêmes que sur données réelles ?**

> **Réponse** : **Globalement oui** pour les hyperparamètres architecturaux (profondeur=5, dimension=256, k=10). Des différences mineures existent pour les paramètres d'entraînement : (1) **learning rate** légèrement plus faible sur réels (5e-4 vs 1e-3) pour éviter l'overfitting ; (2) **dropout** légèrement plus élevé (0.15 vs 0.1) ; (3) **epochs** moins nombreuses (50 vs 100, early stopping). Cette similarité valide l'utilité du pré-entraînement synthétique.

---

**Q : Le dropout de 0.15 semble faible. Avez-vous testé des valeurs plus élevées ?**

> **Réponse** : Oui, nous avons testé dropout ∈ {0, 0.1, 0.15, 0.2, 0.3}. 0.15 est optimal pour notre configuration. Dropout=0.3 dégrade les performances de ~5% (sous-apprentissage), dropout=0 également de ~3% (légère surapprentissage). La valeur relativement faible s'explique par : (1) dataset modeste nécessitant d'exploiter chaque exemple ; (2) régularisation déjà fournie par le pré-entraînement ; (3) autres régularisations actives (weight decay, batch normalization).

---

### 4.3 Évaluation

**Q : Pourquoi une validation croisée 5-fold plutôt que leave-one-out ?**

> **Réponse** : Leave-one-out sur 500 organoïdes nécessiterait 500 entraînements, coûteux en temps (>500 heures). De plus, LOO produit des estimateurs à **forte variance** pour les métriques de classification. La validation croisée 5-fold offre un bon compromis : estimation stable de la performance (~100 exemples par fold de test), temps raisonnable (5 entraînements), et validation standard dans la littérature. Nous répétons la CV sur 3 seeds différentes pour évaluer la variabilité due à l'initialisation.

---

**Q : Avez-vous corrigé pour les comparaisons multiples ?**

> **Réponse** : Pour les comparaisons d'architectures (GCN vs GAT vs EGNN vs DeepSets), nous appliquons une correction de **Bonferroni** aux p-values des tests appariés. Avec 6 comparaisons, le seuil de significativité passe de 0.05 à 0.008. Les différences GAT vs GCN et GAT vs DeepSets restent significatives après correction (p<0.001). La différence GAT vs EGNN n'est pas significative (p~0.03), cohérent avec notre conclusion que les deux sont valables avec des compromis différents.

---

**Q : Comment gérez-vous la variabilité des résultats due à l'initialisation aléatoire ?**

> **Réponse** : Plusieurs stratégies : (1) **Seeds fixées** pour reproductibilité (seed=42 par défaut, testé aussi sur 0, 123) ; (2) **Moyennes sur 3-5 runs** avec seeds différentes pour les résultats finaux ; (3) **Report des écarts-types** dans tous les tableaux de résultats. La variabilité typique est de ±2% en accuracy pour les runs avec seeds différentes, confirmant la stabilité des conclusions. Les comparaisons sont effectuées sur les **mêmes splits** pour éliminer la variabilité due au partitionnement.

---

## 5. Questions sur les limitations

### 5.1 Limitations reconnues

**Q : Pourquoi ne pas avoir développé une approche segmentation-free ?**

> **Réponse** : Une approche segmentation-free (par exemple, PointCloud directement sur voxels d'intensité) a été envisagée mais présente des défis : (1) **volume de données** : des millions de voxels vs des centaines de cellules ; (2) **perte d'abstraction** : le niveau cellulaire est biologiquement pertinent ; (3) **complexité** : les architectures point cloud (PointNet++) sont moins matures que les GNN pour notre tâche. Des approches hybrides (clustering soft, superpixels) sont des perspectives, mais la segmentation explicite reste l'approche la plus interprétable et efficace actuellement.

---

**Q : L'absence de validation inter-laboratoires n'est-ce pas critique ?**

> **Réponse** : Oui, c'est une limitation importante reconnue. Les variations inter-laboratoires (microscopes, protocoles, lignées cellulaires) peuvent induire un **domain shift** significatif. Nos stratégies de mitigation (normalisation d'intensité, équivariance géométrique) atténuent certaines variations mais ne garantissent pas la généralisation. Une **validation multi-sites** est essentielle avant tout déploiement clinique. C'est une priorité pour les collaborations futures dans le cadre d'ANR Morpheus étendu.

---

**Q : L'absence d'analyse d'interprétabilité n'est-elle pas problématique ?**

> **Réponse** : Oui, c'est une limitation reconnue. Sans interprétabilité, on ne peut pas : (1) valider que le modèle utilise des critères biologiquement pertinents ; (2) identifier les cellules/régions discriminantes ; (3) gagner la confiance des biologistes utilisateurs. Des techniques existent (GradCAM sur graphes, analyse des poids d'attention, perturbation analysis) mais n'ont pas été systématiquement appliquées par manque de temps. C'est une **priorité immédiate** pour les travaux futurs, essentielle pour l'adoption par la communauté biologique.

---

**Q : Comment justifiez-vous l'absence de validation statistique des données synthétiques ?**

> **Réponse** : C'est un compromis pragmatique : (1) l'objectif n'était pas de générer des synthétiques "parfaitement réalistes" mais des données utiles pour le pré-entraînement ; (2) le gain empirique de +8% en transfer learning **valide fonctionnellement** l'utilité des synthétiques ; (3) une validation statistique rigoureuse (tests de Kolmogorov-Smirnov, comparaison des fonctions K) nécessite une expertise en statistiques spatiales que nous développons. C'est une amélioration prévue qui renforcerait la rigueur méthodologique.

---

### 5.2 Limitations non mentionnées

**Q : Votre approche fonctionne-t-elle pour des organoïdes en time-lapse ?**

> **Réponse** : Pas directement. Notre pipeline analyse des snapshots statiques. L'extension au time-lapse nécessiterait : (1) **tracking cellulaire** entre frames (problème complexe avec divisions/morts) ; (2) **GNN temporels** (TGNN, TGN) capturant l'évolution ; (3) **volumes de données** considérablement plus importants. C'est une perspective à long terme mentionnée dans la thèse, mais qui dépasse le scope actuel.

---

**Q : Comment gérez-vous les organoïdes fusionnés ou partiellement visibles ?**

> **Réponse** : Le **clustering DBSCAN** sépare automatiquement les organoïdes fusionnés si un espace minimal existe entre eux. Les organoïdes partiellement visibles (coupés par le bord du champ) sont **exclus** car leur graphe serait incomplet. En pratique, le protocole d'imagerie est calibré pour capturer les organoïdes entièrement dans le champ de vue. Pour des acquisitions moins contrôlées, une détection de troncature serait nécessaire.

---

**Q : Que se passe-t-il si un organoïde contient plusieurs phénotypes (mosaïque) ?**

> **Réponse** : Notre modèle produit une prédiction **globale par organoïde**, ne capturant pas les phénotypes mixtes. Un organoïde mosaïque serait classé selon son phénotype dominant (contribution majoritaire au graphe). Pour des applications où les mosaïques sont pertinentes, une approche de **classification par sous-régions** ou de **segmentation sémantique** du graphe serait nécessaire. La prévalence des mosaïques dans nos organoïdes de prostate est faible (<5%), justifiant notre simplification.

---

**Q : Comment votre méthode gère-t-elle les cellules en mitose ou en apoptose ?**

> **Réponse** : Ces cellules sont traitées comme des nœuds ordinaires avec potentiellement des features morphologiques atypiques (taille anormale, forme irrégulière). Cellpose les détecte généralement correctement. Leur présence n'est pas explicitement modélisée car : (1) leur proportion est faible (~1-5%) ; (2) leur distribution spatiale peut être informative (zones prolifératives) et est capturée implicitement par le GNN. Un marquage spécifique (Ki67 pour prolifération, caspase pour apoptose) permettrait une modélisation explicite si pertinent.

---

### 5.3 Scalabilité

**Q : Votre approche scale-t-elle aux très grands organoïdes (>5000 cellules) ?**

> **Réponse** : La scalabilité dépend de l'étape : (1) **Segmentation** : linéaire en nombre de coupes, donc scalable ; (2) **Construction graphe** : O(n log n) avec K-NN, scalable ; (3) **GNN** : O(n×k×L) avec n nœuds, k voisins, L couches - devient coûteux au-delà de ~10,000 cellules. Pour les organoïdes cérébraux (>100,000 cellules), des stratégies de **sampling** (GraphSAINT) ou de **pooling hiérarchique** seraient nécessaires. Notre implémentation actuelle gère confortablement jusqu'à ~5,000 cellules.

---

**Q : Comment gérer un flux continu de milliers d'organoïdes en production ?**

> **Réponse** : Architecture envisagée : (1) **Queue de messages** (RabbitMQ/Kafka) pour les images entrantes ; (2) **Workers de segmentation** parallélisés sur GPU cluster ; (3) **Cache des graphes** pour éviter recalculs ; (4) **Batch inference** GNN optimisé ; (5) **Base de données** des résultats avec API REST. Le goulot d'étranglement reste la segmentation ; avec 10 GPUs, on peut traiter ~700 organoïdes/jour. Une optimisation supplémentaire de Cellpose ou l'utilisation de la méthode ellipses pour pré-filtrage augmenterait ce débit.

---

**Q : Le stockage JSON est-il optimal ? Avez-vous envisagé des formats binaires ?**

> **Réponse** : JSON est choisi pour l'**interprétabilité** et la **portabilité**, pas pour l'efficacité. Pour la production, des formats binaires comme **PyTorch .pt** (serialisation native des Data objects), **HDF5** (compression + accès partiel), ou **Parquet** (colonnes, compression) seraient préférables. Le surcoût JSON est acceptable pour notre échelle (~200 Mo pour 2000 organoïdes) mais deviendrait prohibitif pour >100,000 échantillons. Une migration vers HDF5 est planifiée pour le déploiement.

---

## 6. Questions sur la généralisation

### 6.1 Autres types d'organoïdes

**Q : Avez-vous testé votre pipeline sur d'autres types d'organoïdes ?**

> **Réponse** : Pas de manière formelle. Des tests préliminaires sur des **images publiques d'organoïdes intestinaux** suggèrent que la segmentation et la construction de graphes fonctionnent, mais les performances de classification n'ont pas été évaluées (absence de labels). La transférabilité des poids pré-entraînés sur synthétiques devrait faciliter l'adaptation. Des collaborations sont en cours pour obtenir des datasets annotés d'autres types d'organoïdes.

---

**Q : Quelles adaptations seraient nécessaires pour des organoïdes cérébraux ?**

> **Réponse** : Plusieurs défis spécifiques : (1) **Morphologie** : forme irrégulière non-sphérique nécessitant des processus ponctuels adaptés ; (2) **Hétérogénéité cellulaire** : nombreux types neuronaux nécessitant une segmentation multi-classe et des features spécifiques ; (3) **Taille** : jusqu'à >100,000 cellules nécessitant des stratégies de sampling/pooling ; (4) **Features** : marqueurs neuronaux spécifiques (NeuN, GFAP). L'adaptation nécessiterait 3-6 mois de développement avec expertise en neurobiologie.

---

**Q : Les processus ponctuels sur sphère sont-ils adaptés aux organoïdes tubulaires ?**

> **Réponse** : Non directement. Les organoïdes intestinaux ou rénaux ont des géométries **tubulaires ou ramifiées** mal capturées par une sphère. Des extensions existent : (1) processus ponctuels sur **cylindres** ou **surfaces de révolution** ; (2) processus sur **graphes de squelette** (centerline) ; (3) processus sur **variétés riemanniennes** générales. Ces extensions sont mathématiquement bien définies mais nécessitent un développement spécifique. C'est une perspective de recherche ouverte.

---

### 6.2 Transferabilité

**Q : Votre méthodologie est-elle applicable aux sphéroïdes tumoraux ?**

> **Réponse** : **Oui**, avec des adaptations mineures. Les sphéroïdes tumoraux partagent avec les organoïdes : (1) structure 3D compacte ; (2) organisation cellulaire significative (gradient nécrose/prolifération) ; (3) phénotypes morphologiques pertinents. Les différences sont : (1) absence de différenciation tissulaire organisée ; (2) focus sur viabilité/croissance plutôt que architecture. Le pipeline s'applique directement ; les tâches de prédiction (réponse au traitement) nécessiteraient un re-entraînement sur données spécifiques.

---

**Q : Les GNN équivariants sont-ils vraiment nécessaires pour toutes les applications ?**

> **Réponse** : Pas nécessairement. L'équivariance E(3) est **cruciale** quand : (1) les données n'ont pas d'orientation canonique (organoïdes en suspension) ; (2) le dataset est petit (l'équivariance réduit le besoin en augmentation). Pour des applications avec orientation fixe (coupes histologiques 2D, tissus sur lame) ou avec beaucoup de données, des GNN standards avec augmentation peuvent suffire. L'équivariance apporte une **garantie théorique** mais a un coût computationnel ; le choix dépend du contexte.

---

### 6.3 Domain adaptation

**Q : Comment adapter le modèle à un nouveau laboratoire ?**

> **Réponse** : Protocole recommandé : (1) **Vérification** de la segmentation sur ~50 images (fine-tune Cellpose si nécessaire) ; (2) **Normalisation** des intensités avec les mêmes percentiles ; (3) **Fine-tuning** du GNN sur ~50-100 exemples annotés du nouveau laboratoire. Si le domain shift est important, des techniques de **domain adaptation** (DANN, CORAL) peuvent aligner les distributions de features. Nous estimons le coût à ~1 semaine de travail pour l'adaptation à un nouveau site.

---

**Q : Combien d'exemples annotés pour un nouveau type d'organoïde ?**

> **Réponse** : Estimation basée sur nos courbes d'apprentissage : (1) **Minimum viable** : ~50-100 exemples pour un fine-tuning basique depuis pré-entraînement synthétique ; (2) **Performance acceptable** (~80% accuracy) : ~200-300 exemples ; (3) **Performance optimale** : ~500+ exemples avec variation suffisante. Ces chiffres supposent que le nouveau type partage des patterns spatiaux avec les organoïdes de prostate ; pour des structures très différentes (organoïdes cérébraux), il faudrait possiblement 2-3× plus.

---

## 7. Questions sur les perspectives

### 7.1 Court terme

**Q : Quelle serait la priorité entre les perspectives mentionnées ?**

> **Réponse** : Par ordre de priorité : (1) **Analyse d'interprétabilité** - essentielle pour l'adoption par les biologistes, réalisable en 2-3 mois ; (2) **Validation multi-sites** - critique pour la crédibilité clinique, dépend de collaborations ; (3) **Interface utilisateur** - pour démocratiser l'outil, 3-6 mois de développement ; (4) **Intégration multi-modale** - potentiel scientifique élevé mais nécessite des données spécifiques. La prédiction thérapeutique est passionnante mais requiert des cohortes cliniques substantielles.

---

**Q : L'intégration de données multi-modales est-elle réaliste techniquement ?**

> **Réponse** : **Oui**, les briques existent : (1) les technologies de transcriptomique spatiale (Visium, MERFISH) produisent des données avec coordonnées spatiales directement intégrables comme features de nœuds ; (2) des architectures de fusion multi-modale (attention cross-modale, early/late fusion) sont bien établies ; (3) le framework PyG supporte des features de dimension arbitraire. Le défi principal est l'**acquisition de données appariées** (même organoïde imagé ET séquencé), coûteuse et techniquement complexe.

---

**Q : Les alpha-shapes, pourquoi ne pas les avoir déjà implémentés ?**

> **Réponse** : Les alpha-shapes pour la caractérisation morphologique globale sont une **perspective identifiée tardivement** lors de la rédaction. L'implémentation technique est relativement simple (scipy.spatial, CGAL) mais la **validation biologique** (corrélation avec phénotypes, choix du paramètre α optimal) nécessite une étude dédiée. C'est une extension naturelle du pipeline actuel, réalisable en 1-2 mois, qui enrichirait la représentation avec des features de forme globale complémentaires aux patterns locaux capturés par le GNN.

---

### 7.2 Long terme

**Q : La prédiction de réponse thérapeutique nécessite-t-elle un modèle différent ?**

> **Réponse** : L'architecture de base (GNN sur graphes cellulaires) reste pertinente, mais des adaptations sont nécessaires : (1) **Architecture siamoise** comparant pré/post-traitement ; (2) **Features enrichies** capturant les changements induits (viabilité, marqueurs d'apoptose) ; (3) **Modélisation temporelle** si time-lapse disponible. Le changement fondamental est la **tâche** : prédire une réponse clinique (binaire ou continue) plutôt qu'un phénotype morphologique. Les principes méthodologiques (graphes, équivariance, transfer learning) restent valables.

---

**Q : Les modèles génératifs de graphes sont-ils matures ?**

> **Réponse** : Ils progressent rapidement mais restent **moins matures** que leurs équivalents pour images. Les approches récentes (Graph Diffusion, GraphVAE, GraphRNN) produisent des graphes plausibles mais avec des limitations : (1) génération de graphes de grande taille (>1000 nœuds) encore difficile ; (2) contrôle des propriétés globales moins précis que les processus ponctuels ; (3) validation de réalisme complexe. Pour nos besoins actuels (pré-entraînement), les processus ponctuels suffisent ; les génératifs appris deviendraient pertinents pour de l'augmentation de données sophistiquée.

---

**Q : Quel horizon temporel pour une application clinique réelle ?**

> **Réponse** : Estimation réaliste : (1) **Recherche** (current → 2 ans) : validation multi-sites, études prospectives pilotes ; (2) **Développement** (2-4 ans) : interface clinique, intégration workflow, études réglementaires ; (3) **Certification** (4-6 ans) : essais cliniques, certification IVDR/FDA ; (4) **Déploiement** (6+ ans) : adoption progressive. Ce timeline est comparable aux dispositifs médicaux basés sur IA récents. Des applications non-cliniques (criblage pharma, recherche) sont réalisables à plus court terme (1-2 ans).

---

### 7.3 Impact

**Q : Quel modèle économique pour le déploiement ?**

> **Réponse** : Plusieurs options : (1) **Open-source académique** : code libre, adoption communautaire, valorisation indirecte (publications, collaborations) - notre choix actuel ; (2) **SaaS** : plateforme cloud avec tarification à l'usage pour pharma/biotech ; (3) **Licence** : vente de licences à des entreprises d'imagerie (Zeiss, Leica) pour intégration ; (4) **Spin-off** : création d'une startup avec levée de fonds. Le choix dépend des objectifs et opportunités ; la voie open-source favorise l'impact scientifique, le SaaS la pérennité économique.

---

**Q : Comment voyez-vous l'adoption par la communauté biologique ?**

> **Réponse** : L'adoption sera **progressive** et conditionnée par : (1) **Accessibilité** : interface simple, documentation claire, pas de compétences en programmation requises ; (2) **Validation** : publications dans des journaux biologiques (pas seulement informatiques), témoignages d'utilisateurs ; (3) **Interprétabilité** : les biologistes doivent comprendre pourquoi le modèle fait ses prédictions ; (4) **Support** : tutoriels, workshops, communauté active. L'expérience de Cellpose (>5000 citations, adoption massive) montre qu'un outil bien conçu peut transformer les pratiques.

---

## 8. Questions épistémologiques et transversales

### 8.1 Choix méthodologiques

**Q : Pourquoi la classification supervisée plutôt que le clustering non supervisé ?**

> **Réponse** : La classification supervisée était appropriée car : (1) les phénotypes cystique/choux-fleurs sont **biologiquement définis** et cliniquement pertinents ; (2) des labels experts existaient (même bruités) ; (3) l'objectif était un outil de **prédiction opérationnel**. Le clustering non supervisé reste pertinent pour : découvrir de nouveaux phénotypes, explorer les 1,772 organoïdes à labels incertains, valider la pertinence biologique des classes. Les deux approches sont complémentaires.

---

**Q : N'y a-t-il pas un risque de circularité avec les labels basés sur la morphologie ?**

> **Réponse** : Le risque existe mais est **limité** : (1) les labels batch sont assignés selon les **conditions de culture**, pas l'observation morphologique individuelle ; (2) notre modèle apprend des **patterns spatiaux fins** (organisation cellulaire) au-delà de la forme globale visible à l'œil ; (3) la sélection des 500 organoïdes "bien différenciés" élimine les cas ambigus. La vraie validation serait de corréler nos prédictions avec des outcomes biologiques indépendants (expression génique, réponse à traitement), prévu pour les travaux futurs.

---

**Q : Comment définissez-vous biologiquement la frontière entre cystique et choux-fleurs ?**

> **Réponse** : La frontière est définie par le **phénotype architectural** : (1) **Cystique** : cavité centrale (lumen) bordée d'un épithélium polarisé, surface lisse et sphérique, distribution cellulaire uniforme ; (2) **Choux-fleurs** : absence de lumen organisé, surface irrégulière avec protubérances, agrégats cellulaires denses. Biologiquement, cela reflète une **différenciation normale vs perturbée** de l'épithélium prostatique. La frontière est un **continuum** ; les 500 organoïdes sélectionnés représentent les extrêmes bien caractérisés.

---

### 8.2 Reproductibilité

**Q : Le code est-il reproductible ?**

> **Réponse** : **Oui**, avec les précautions suivantes : (1) **Seeds fixées** (numpy, torch, random) pour reproductibilité exacte ; (2) **Versions lockées** des dépendances (requirements.txt avec versions exactes) ; (3) **Fichiers de configuration** YAML pour tous les hyperparamètres ; (4) **Scripts de bout en bout** recréant les résultats du papier ; (5) **Documentation** détaillée. Les résultats sont reproductibles à <1% de variance sur la même machine ; des différences mineures peuvent apparaître entre GPU différentes (opérations non-déterministes de cuDNN).

---

**Q : Les résultats sont-ils reproductibles sur d'autres GPU/versions de PyTorch ?**

> **Réponse** : **Globalement oui** avec quelques nuances : (1) la **tendance** des résultats (GAT > GCN, +8% transfer learning) est robuste ; (2) les **valeurs exactes** peuvent varier de ±1-2% entre configurations ; (3) certaines opérations CUDA sont non-déterministes par défaut (atomicAdd) - on peut forcer le déterminisme avec torch.use_deterministic_algorithms(True) au prix de ~10% de ralentissement. Nous avons testé sur RTX 3080, A100, et V100 avec PyTorch 1.12-2.0, résultats cohérents.

---

### 8.3 Interprétabilité

**Q : Comment un biologiste peut-il comprendre pourquoi le modèle fait une prédiction ?**

> **Réponse** : Actuellement, **difficilement** - c'est une limitation reconnue. Les pistes pour l'interprétabilité incluent : (1) **Visualisation des embeddings** (UMAP/t-SNE) montrant le clustering des organoïdes ; (2) **Poids d'attention** de GAT identifiant les connexions importantes ; (3) **GradCAM sur graphes** localisant les régions discriminantes ; (4) **Perturbation analysis** mesurant l'impact de la suppression de nœuds/arêtes. Ces techniques sont implémentables ; leur validation biologique avec des experts est la prochaine étape.

---

**Q : Les poids d'attention de GAT sont-ils interprétables biologiquement ?**

> **Réponse** : **Potentiellement**, mais avec prudence. Les poids d'attention indiquent quels voisins influencent le plus l'embedding d'un nœud, mais : (1) ils varient par couche et par tête d'attention ; (2) ils peuvent capturer des corrélations non causales ; (3) leur interprétation nécessite une expertise biologique. Des analyses préliminaires suggèrent que les cellules en périphérie et aux interfaces cluster/non-cluster reçoivent plus d'attention, cohérent avec l'intuition biologique, mais une validation formelle est nécessaire.

---

### 8.4 Éthique et réglementation

**Q : Quelles implications éthiques de l'automatisation ?**

> **Réponse** : Plusieurs dimensions : (1) **Positive** : réduction de l'expérimentation animale (principe des 3R), démocratisation de l'expertise, reproductibilité accrue ; (2) **Vigilance** : risque de sur-confiance dans les prédictions automatisées, nécessité de validation humaine pour décisions cliniques, biais potentiels des données d'entraînement ; (3) **Transparence** : les utilisateurs doivent comprendre les limitations et incertitudes du modèle. L'outil doit être présenté comme aide à la décision, pas comme oracle infaillible.

---

**Q : Comment envisagez-vous la certification comme dispositif médical ?**

> **Réponse** : Processus en plusieurs étapes : (1) **Classification** du dispositif (probablement classe IIa selon IVDR pour aide au diagnostic) ; (2) **Dossier technique** : validation analytique, performances cliniques, gestion des risques ; (3) **Système qualité** ISO 13485 ; (4) **Études cliniques** prospectives sur cohortes suffisantes ; (5) **Marquage CE** via organisme notifié. Timeline estimé : 3-5 ans, coût : 500k-2M€. Une collaboration avec un partenaire industriel expérimenté serait nécessaire.

---

**Q : Quelle responsabilité en cas d'erreur de diagnostic ?**

> **Réponse** : Question juridique complexe dépendant du contexte d'utilisation : (1) **Outil de recherche** : responsabilité du chercheur utilisateur ; (2) **Dispositif médical certifié** : responsabilité partagée fabricant/utilisateur selon les usages prévus ; (3) **Aide à la décision** : la décision finale reste au clinicien qui porte la responsabilité. Notre position : l'outil fournit une **information** (probabilité de phénotype), pas une **décision** ; la responsabilité de l'interprétation et de l'action incombe à l'utilisateur qualifié.

---

## 9. Questions techniques pointues

### 9.1 Sur les GNN

**Q : Expliquez formellement la différence entre invariance et équivariance dans EGNN.**

> **Réponse** : Soit T une transformation (rotation, translation) et f un modèle.
> - **Invariance** : f(T(x)) = f(x) → la sortie est **identique** quelle que soit la transformation de l'entrée. Exemple : la prédiction du phénotype ne doit pas changer si on tourne l'organoïde.
> - **Équivariance** : f(T(x)) = T(f(x)) → la sortie se **transforme de manière cohérente** avec l'entrée. Exemple : si on tourne l'organoïde, les embeddings des cellules tournent aussi.
> 
> Dans EGNN, les **features scalaires** (embeddings) sont invariantes, les **coordonnées** sont équivariantes. La prédiction finale (après pooling) est invariante.

---

**Q : Comment EGNN garantit-il l'équivariance E(3) dans la mise à jour des coordonnées ?**

> **Réponse** : EGNN met à jour les coordonnées par :
> 
> x_i^{l+1} = x_i^l + Σ_j (x_i^l - x_j^l) · φ(m_ij)
> 
> où φ est une fonction scalaire des messages invariants m_ij. La clé est que (x_i - x_j) est un **vecteur équivariant** (il tourne avec les coordonnées) et φ(m_ij) est **invariant** (scalaire ne dépendant que de distances et features invariantes). Le produit vecteur × scalaire reste équivariant. Les translations sont gérées en utilisant uniquement des différences de coordonnées (x_i - x_j), jamais les coordonnées absolues.

---

**Q : Comment fonctionne le mécanisme d'attention multi-têtes dans GAT ?**

> **Réponse** : Pour chaque arête (i,j) et tête d'attention k :
> 1. **Projection** : q_i^k = W_q^k · h_i, k_j^k = W_k^k · h_j, v_j^k = W_v^k · h_j
> 2. **Score d'attention** : e_ij^k = LeakyReLU(a^k · [q_i^k || k_j^k])
> 3. **Normalisation softmax** : α_ij^k = softmax_j(e_ij^k)
> 4. **Agrégation** : h_i' = || Σ_j α_ij^k · v_j^k (concaténation des K têtes)
> 
> Les têtes multiples permettent d'apprendre différents types de relations (proximité, similarité de features, etc.) en parallèle.

---

**Q : Qu'est-ce que l'over-smoothing et comment l'avez-vous évité ?**

> **Réponse** : L'over-smoothing est le phénomène où, après plusieurs couches de message passing, les embeddings de tous les nœuds convergent vers des valeurs similaires, perdant leur pouvoir discriminant. Il se produit car chaque couche moyenne les représentations des voisins.
> 
> **Nos stratégies d'évitement** :
> 1. **Connexions résiduelles** : h_i^{l+1} = h_i^{l+1} + 0.2 · h_i^l préserve l'information initiale
> 2. **Dropout d'arêtes** : évite la sur-propagation
> 3. **Batch normalization** : re-centre et re-scale les embeddings
> 4. **Profondeur limitée** : 5 couches (optimum empirique)

---

### 9.2 Sur les processus ponctuels

**Q : Pouvez-vous détailler le processus de Matérn cluster utilisé ?**

> **Réponse** : Le processus de Matérn cluster sur la sphère se génère en deux étapes :
> 1. **Parents** : N_p ∼ Poisson(λ_p · Aire_sphère) points uniformément distribués sur la sphère (centres de clusters)
> 2. **Enfants** : pour chaque parent p, générer N_c ∼ Poisson(λ_c) points dans un disque géodésique de rayon r centré en p, avec distribution uniforme sur le disque
> 
> **Paramètres** :
> - λ_p : intensité des parents (contrôle le nombre de clusters)
> - λ_c : intensité des enfants par cluster
> - r : rayon des clusters
> 
> Le nombre total de points est N = N_p · E[N_c] = λ_p · λ_c · Aire. Quand λ_p → ∞ et r → 0 avec N fixe, on retrouve le Poisson homogène.

---

**Q : Comment avez-vous adapté les statistiques de Ripley à la géométrie sphérique ?**

> **Réponse** : Les fonctions K, F, G classiques sont définies pour des domaines euclidiens. L'adaptation à la sphère nécessite :
> 1. **Distance géodésique** au lieu de distance euclidienne : d(p,q) = arccos(p·q) pour points sur sphère unité
> 2. **Correction de bord sphérique** : pas de bord sur une sphère fermée, mais les estimateurs doivent utiliser le volume correct (4π/3 pour K)
> 3. **Théorie de référence** : K(r) = 2π(1 - cos(r)) pour Poisson homogène sur la sphère (vs πr² en 2D euclidien)
> 
> Nous utilisons l'implémentation de la librairie `spatstat` adaptée aux domaines sphériques.

---

**Q : Quelle est la différence entre processus de Matérn et de Thomas ?**

> **Réponse** : Les deux sont des **processus de cluster** avec parents et enfants, mais diffèrent par la distribution des enfants autour des parents :
> - **Matérn** : enfants uniformément distribués dans un disque de rayon r autour du parent → clusters à bords nets
> - **Thomas** : enfants distribués selon une gaussienne centrée sur le parent (écart-type σ) → clusters à bords flous
> 
> Thomas est plus "naturel" pour des processus biologiques avec diffusion, Matérn est plus simple et contrôlé. Pour notre application, les deux produisent des résultats similaires ; nous avons choisi Matérn pour sa simplicité d'interprétation.

---

### 9.3 Sur la segmentation

**Q : Comment fonctionne exactement le flow tracking de Cellpose ?**

> **Réponse** : Cellpose prédit pour chaque pixel un **vecteur de gradient** pointant vers le centre de sa cellule. Le flow tracking fonctionne ainsi :
> 1. **Prédiction** : le réseau prédit (∂x, ∂y, ∂z) pour chaque voxel + probabilité foreground/background
> 2. **Intégration** : depuis chaque pixel foreground, on suit le champ de gradient par intégration numérique (Euler) sur ~200 itérations
> 3. **Convergence** : les pixels d'une même cellule convergent vers le même point (centre approximatif)
> 4. **Clustering** : les pixels sont groupés par leur point de convergence → une cellule par cluster
> 
> L'élégance est que le réseau n'a pas besoin de prédire des frontières explicites, seulement des directions.

---

**Q : Qu'est-ce que la knowledge distillation et comment l'avez-vous appliquée ?**

> **Réponse** : La knowledge distillation transfère les "connaissances" d'un modèle teacher (grand, performant) vers un student (compact, rapide) :
> 
> **Notre application** :
> 1. **Teacher** : Cellpose cyto2 complet (frozen)
> 2. **Student** : FastCellpose (50% moins de canaux)
> 3. **Loss** : L = 0.3 · L_hard(student, ground_truth) + 0.7 · L_soft(student, teacher)
>    - L_hard : perte classique sur vérité terrain
>    - L_soft : KL-divergence entre distributions de probabilité student et teacher
> 
> Les soft labels du teacher contiennent plus d'information (incertitudes, frontières floues) que les hard labels binaires, permettant au student d'apprendre plus efficacement.

---

**Q : Le pruning de 30% des poids, comment avez-vous déterminé ce seuil ?**

> **Réponse** : Par **recherche empirique** : nous avons testé pruning ∈ {10%, 20%, 30%, 40%, 50%} et mesuré l'impact sur F1-score après fine-tuning :
> - 10% : F1 = 0.96 (aucune dégradation), speedup 1.05×
> - 20% : F1 = 0.96, speedup 1.15×
> - **30%** : F1 = 0.96, speedup **1.25×** ← meilleur compromis
> - 40% : F1 = 0.94, speedup 1.35×
> - 50% : F1 = 0.91, speedup 1.45×
> 
> Le pruning L1-unstructured supprime les poids de plus faible magnitude. 30% est le seuil où la redondance est éliminée sans impacter la représentation. Le fine-tuning post-pruning (10 epochs) récupère les éventuelles micro-pertes.

---

## 10. Questions provocantes / Devil's advocate

**Q : Si les statistiques spatiales sont plus robustes au bruit, pourquoi ne pas les utiliser ?**

> **Réponse** : La robustesse au bruit n'est qu'un critère parmi d'autres. Les statistiques spatiales **échouent** sur le critère décisif pour notre application : la **généralisation géométrique**. Les organoïdes réels ont des formes variables (pas des sphères parfaites), et les statistiques spatiales chutent de 35% d'accuracy sur les ellipsoïdes. Les GNN maintiennent 82%. Pour une application réelle avec données hétérogènes, les GNN sont donc le bon choix malgré leur sensibilité accrue au bruit (qu'on peut mitiger par la qualité de segmentation et l'augmentation).

---

**Q : N'est-ce pas de la sur-ingénierie ? Un CNN 2D sur projections ne suffirait-il pas ?**

> **Réponse** : Nous avons testé : un CNN 2D sur MIP (Maximum Intensity Projection) atteint ~70% d'accuracy. La perte de 14% par rapport à notre approche (84%) s'explique par la **destruction d'information 3D** : les relations de voisinage en Z, l'organisation en couches, la structure du lumen sont perdues dans la projection. Pour une tâche simple (présence/absence d'organoïde), un CNN 2D suffirait ; pour la classification fine de phénotypes basés sur l'architecture 3D, les GNN sont justifiés.

---

**Q : 84% d'accuracy avec 500 exemples, n'importe quel ML classique n'atteindrait-il pas ce niveau ?**

> **Réponse** : Nous avons comparé : Random Forest sur descripteurs globaux atteint 72%, soit 12% de moins. La différence est significative et s'explique par la capacité des GNN à exploiter la **structure relationnelle locale** sans feature engineering. De plus, notre 84% est obtenu grâce au transfer learning depuis synthétiques ; sans celui-ci (from scratch), on atteint 76% - toujours supérieur au RF mais l'écart se réduit. Les GNN sont particulièrement avantageux quand on dispose de données synthétiques pour le pré-entraînement.

---

**Q : Les GNN ne sont-ils pas une mode passagère ?**

> **Réponse** : Les GNN répondent à un **besoin fondamental** : traiter des données structurées en graphes. Ce besoin existait avant les GNN (avec des méthodes moins expressives comme les kernel de graphes) et persistera. L'adoption massive en chimie computationnelle (AlphaFold utilise des composants GNN), réseaux sociaux, recommandation, et maintenant biologie suggère une **maturité croissante** plutôt qu'une mode éphémère. Les architectures évolueront (Graph Transformers ?), mais le paradigme de traitement de données relationnelles restera pertinent.

---

**Q : Votre contribution est-elle vraiment originale ou une simple application ?**

> **Réponse** : Les contributions originales incluent : (1) **Première application** systématique des GNN géométriques aux organoïdes 3D ; (2) **Génération synthétique** par processus ponctuels spatiaux pour le pré-entraînement, approche inédite ; (3) **Faster Cellpose** optimisé par distillation ; (4) **Étude comparative** quantitative statistiques spatiales vs GNN (GRETSI 2025) ; (5) **Pipeline intégré** de bout en bout. L'originalité réside dans la **combinaison** et l'**adaptation** de méthodes existantes à un problème nouveau, plus que dans l'invention de nouvelles architectures. C'est une contribution d'**ingénierie scientifique** plutôt que de recherche fondamentale en ML.

---

**Q : Si la segmentation est si critique, pourquoi ne pas avoir fait une thèse dessus ?**

> **Réponse** : La segmentation cellulaire est un problème **largement résolu** par Cellpose et alternatives - notre contribution (Faster Cellpose) est une **optimisation**, pas une innovation fondamentale. L'apport scientifique principal de cette thèse est ailleurs : la **représentation par graphes**, les **GNN géométriques**, la **génération synthétique**, le **transfer learning**. La segmentation est une brique nécessaire mais commoditisée ; la valeur ajoutée est dans l'exploitation intelligente de ses résultats.

---

**Q : N'aurait-il pas été plus utile de collecter plus de données réelles ?**

> **Réponse** : Collecter 10× plus de données (5,000 organoïdes annotés) aurait coûté ~2 ans de travail expérimental supplémentaire et ~100k€ en temps expert. Notre stratégie de génération synthétique + transfer learning atteint des performances comparables avec seulement 500 exemples réels. C'est une approche **pragmatique** maximisant le rapport impact/coût. De plus, les principes méthodologiques développés (génération, transfer) sont **transférables** à d'autres domaines avec peu de données, amplifiant l'impact au-delà de notre application spécifique.

---

## Conseils pour répondre aux questions

### Stratégies générales
1. **Écouter complètement** la question avant de répondre
2. **Reformuler** si nécessaire pour montrer la compréhension
3. **Structurer** la réponse (contexte → réponse → conclusion)
4. **Reconnaître** les limitations quand elles sont légitimes
5. **Proposer** des pistes de solution pour les points faibles

### Phrases utiles
- "C'est une excellente question qui touche à une limitation que nous avons identifiée..."
- "Nous avons effectivement envisagé cette approche, mais..."
- "C'est une perspective intéressante pour des travaux futurs..."
- "Les résultats préliminaires suggèrent que..., mais une validation plus approfondie serait nécessaire"

### Points à ne pas oublier
- Toujours ramener aux **contributions** de la thèse
- Mentionner les **publications** associées (GRETSI 2025)
- Souligner le caractère **interdisciplinaire** du travail
- Rappeler le contexte **collaboratif** (ANR Morpheus, partenaires)
