# Defense Speech — English Speaking Text

**Thesis**: Deep Learning for Organoid Analysis: Graph-Based Modeling of 3D Cellular Architectures  
**Candidate**: Alexandre Martin  
**Duration**: 45 minutes

---

## PART 1: INTRODUCTION (6-7 min)

### Slide 1: Title (30 sec)

Good morning everyone, and thank you for being here today.

I am Alexandre Martin, and I am honored to present my doctoral thesis entitled "Deep Learning for Organoid Analysis: Graph-Based Modeling of 3D Cellular Architectures."

This work was conducted at INRIA Sophia-Antipolis within the Morpheme team, in collaboration with IPMC Nice and Paris Cité University, as part of the ANR Morpheus project.

---

### Slide 2: Hook — Organoids Revolutionize Biology (1 min)

Let me start by introducing organoids. Organoids are often called "mini-organs in a dish." They are three-dimensional structures grown in vitro from stem cells that self-organize to mimic the architecture and function of real organs.

Unlike traditional two-dimensional cell cultures, where cells grow on flat surfaces and lose their natural organization, organoids reproduce the spatial complexity we observe in vivo: they form layers, develop internal cavities called lumens, and exhibit cellular polarization.

Since their first description for intestinal tissue in 2009 by Hans Clevers' team, organoids have been developed for many organs: brain, kidney, liver, lung, pancreas, and many others. They bridge the gap between oversimplified 2D cultures and expensive, ethically problematic animal models.

Their applications are transformative: personalized medicine, drug screening, disease modeling, and potentially regenerative medicine.

---

### Slide 3: Our Prostate Organoids (1 min)

In this thesis, we focus on prostate organoids. We observe two main phenotypes that are our classification targets.

The first is the **cystic phenotype**, which represents healthy organoids. These have a spherical shape with a central cavity, organized cellular structure, and smooth surface. At the cellular level, cells are distributed relatively uniformly.

The second is the **cauliflower phenotype**, which represents a perturbed state. These organoids have an irregular surface with multiple buds, disorganized structure, and at the cellular level, cells form local clusters with high spatial aggregation.

Our dataset, collected over 22 months through collaboration with IPMC Nice and Paris Cité, contains 2,272 individual organoids extracted from 1,311 imaged samples. This represents a substantial resource for our domain.

---

### Slide 4: Scientific Challenge #1 — Automatic Quantification (45 sec)

Why do we need automated analysis? Let me explain the first major challenge.

Currently, expert biologists analyze organoids manually. This takes 15 to 30 minutes per organoid. For a study involving thousands of organoids — which is typical for drug screening — this simply doesn't scale.

Moreover, manual analysis is subjective. Different experts may classify the same organoid differently. Even the same expert may give different assessments at different times. This variability undermines reproducibility.

We need automated, objective, reproducible analysis tools.

---

### Slide 5: Scientific Challenge #2 — Rare Annotated Data (45 sec)

The second challenge is the scarcity of annotated data.

Unlike computer vision where we have ImageNet with 14 million labeled images, the organoid domain has no public datasets. Expert annotation is expensive — a biologist's time costs around 50 euros per hour. Getting thousands of annotations is simply not practical.

This data scarcity is a fundamental bottleneck for training deep learning models, which typically require large amounts of labeled examples.

---

### Slide 6: Scientific Challenge #3 — Geometric Robustness (45 sec)

The third challenge is geometric robustness.

Organoids cultured in 3D suspension have no preferred orientation. Their position and rotation in the culture dish are essentially random. This means our predictions must be invariant to rotations and translations — the same organoid oriented differently should receive the same classification.

Additionally, acquisition conditions vary between laboratories — different microscopes, different protocols, different imaging parameters. Our methods must be robust to these variations.

---

### Slide 7: Research Questions (1 min)

These challenges lead us to three core research questions that structure this thesis.

**Question 1**: Are geometric graphs an effective representation for organoids in machine learning? Can we capture the essential biological information while dramatically compressing the data?

**Question 2**: Do equivariant Graph Neural Networks outperform classical approaches? Do they provide the geometric robustness we need?

**Question 3**: Is transfer learning from synthetic data effective? Can we generate realistic synthetic organoids to compensate for the lack of real annotations?

---

### Slide 8: Presentation Outline (30 sec)

Here is the outline of my presentation.

First, I will present the **theoretical foundations**: the theory of graphs and Graph Neural Networks, explaining why this representation is natural for organoids.

Second, I will describe our **methodology**: the complete pipeline from raw images to predictions, including our contributions to segmentation and synthetic data generation.

Third, I will present our **experimental results** on both synthetic and real data, demonstrating the effectiveness of our approach.

Finally, I will conclude with the **contributions and perspectives** for future work.

---

## PART 2: THEORETICAL FOUNDATIONS (12-14 min)

### Slide 9: Part 2 Title (10 sec)

Let me now turn to the theoretical foundations of our approach: graphs and Graph Neural Networks.

---

### Slide 10: Limitations of 3D CNNs (1.5 min)

A natural question is: why not use 3D Convolutional Neural Networks, which have been successful for medical image analysis?

The answer lies in practical constraints. A single organoid imaged at high resolution generates about 2 gigabytes of data — 2048 by 2048 pixels, times 200 slices in Z. Loading this into GPU memory for a 3D CNN is prohibitive.

The standard solution is downsampling — reducing resolution to fit in memory. But this destroys the fine cellular details we need. Individual cells become unresolvable.

Furthermore, 3D CNNs don't have native geometric invariance. They're not naturally invariant to 3D rotations. Achieving this requires extensive data augmentation, which is computationally expensive and doesn't provide guarantees.

Finally, CNNs treat the organoid as a grid of voxels. They don't explicitly model cells as individual entities or capture the relational structure — which cell is next to which.

---

### Slide 11: Graph Representation — A Natural Abstraction (1.5 min)

This leads us to our key insight: graphs provide a natural abstraction for organoids.

Think about it: cells within an organoid form a network of spatial interactions. They contact each other, communicate through signaling, coordinate their behaviors. This relational organization — who is next to whom — largely determines the organoid's phenotype.

Our abstraction is simple: each cell becomes a node in a graph. Spatial proximity defines edges — neighboring cells are connected. Node features capture cellular properties: 3D position, volume, morphology.

This representation offers spectacular compression. We go from gigabytes of raw image data to megabytes of graph data — a factor of 1000. Yet we preserve the biologically relevant structural information.

On this slide you can see the transformation: raw 3D image, to point cloud of cell centroids, to geometric graph with edges encoding neighborhood relationships.

---

### Slide 12: Introduction to Graph Neural Networks (2 min)

Given this graph representation, how do we learn from it? This is where Graph Neural Networks come in.

The key challenge is that graphs are non-Euclidean data. Unlike images where pixels lie on a regular grid, nodes in a graph have variable numbers of neighbors, no canonical ordering, no fixed structure.

GNNs solve this through the **message passing paradigm**. The idea is elegant: each node iteratively aggregates information from its neighbors to update its representation.

Let me explain with this animation. At layer zero, each node has its initial features. At layer one, each node collects messages from its neighbors and updates. At layer two, information has propagated further. After L layers, each node's representation captures information from its L-hop neighborhood.

Mathematically, the update rule is: the new representation of node i equals an UPDATE function applied to its current representation and an AGGREGATE of its neighbors' representations.

This is analogous to convolution on images, but generalized to arbitrary graph structures.

---

### Slide 13: Global Pooling and Classification (1 min)

For organoid classification, we need a graph-level prediction, not node-level.

This requires **global pooling**: aggregating all node representations into a single graph representation. The simplest approaches are mean pooling — averaging all node features — or max pooling — taking the element-wise maximum.

More sophisticated approaches include attention-weighted pooling, where we learn which nodes are most important for the prediction.

Once we have this graph-level representation, we apply a classification head — typically a multi-layer perceptron — to produce class probabilities.

---

### Slide 14: GCN Architecture — Baseline (1 min)

Let me now describe the specific architectures we evaluated, starting with our baseline.

The **Graph Convolutional Network**, or GCN, was introduced by Kipf and Welling in 2017. It performs normalized mean aggregation: each node's update is a weighted average of its neighbors' features, normalized by the degrees.

The formula shows this: we sum over neighbors, divide by the square root of degrees for normalization, and apply a learnable weight matrix.

GCN is simple and effective, but it treats all neighbors equally. It doesn't distinguish whether a neighbor is close or far, important or unimportant.

---

### Slide 15: GAT Architecture — Our Main Choice (1.5 min)

The **Graph Attention Network**, or GAT, addresses this limitation through attention.

The key idea is to learn attention coefficients that weight the importance of each neighbor. Not all neighbors contribute equally to the prediction — some are more informative than others.

The attention mechanism works as follows: for each pair of connected nodes, we compute an attention score using a learned attention vector. We apply a LeakyReLU nonlinearity and softmax to get normalized weights. Then we aggregate neighbor features weighted by these learned coefficients.

GAT also uses **multi-head attention** — multiple independent attention mechanisms whose outputs are concatenated. This allows capturing different types of relationships simultaneously.

As we'll see in the results, GAT achieves the best performance among the architectures we tested, precisely because of this adaptive weighting mechanism.

---

### Slide 16: DeepSets Architecture — Comparison Baseline (1 min)

As an alternative baseline, we also evaluated **DeepSets**, a set-based approach.

DeepSets treats the organoid as an unordered set of cells, without explicit neighborhood structure. Each cell is encoded independently by a neural network phi, then all encodings are aggregated globally — typically by summation or averaging — and finally processed by another network rho.

This approach is invariant to permutations: the order in which we process cells doesn't matter. But it completely ignores local spatial structure. Each cell is treated in isolation before the global aggregation.

Surprisingly, DeepSets performs reasonably well — this tells us that global statistics about cellular properties are quite informative. But as we'll see, architectures that explicitly model local structure perform better.

---

### Slide 17: Geometric Equivariance — Key Concept (1.5 min)

Now let me introduce a concept that is crucial for organoid analysis: geometric equivariance.

**Why is this important for organoids?** As I mentioned earlier, organoids in 3D culture have no preferred orientation. They float randomly. The biological phenotype doesn't depend on how we happen to orient the organoid when imaging.

This means our model's prediction should be **invariant** to rotations and translations: rotate the organoid, get the same prediction.

Let me clarify the distinction between invariance and equivariance, which are related but different concepts.

**Invariance** means the output is identical regardless of transformation: f of T of x equals f of x. For classification, we want invariance — the class doesn't change if we rotate.

**Equivariance** means the output transforms coherently: f of T of x equals T of f of x. For intermediate representations — like node embeddings — equivariance is appropriate. The representation should rotate along with the input.

---

### Slide 18: The E(3) Group (1 min)

The relevant symmetry group for 3D biological structures is **E(3)** — the Euclidean group in 3 dimensions.

E(3) encompasses three types of transformations:
- **Rotations**: any 3D rotation around any axis
- **Translations**: shifting the entire structure in space  
- **Reflections**: mirror symmetries

For organoid analysis, all these transformations should leave our predictions unchanged. An organoid and its mirror image, rotated and translated, should receive the same classification.

The key insight is that we can build architectures that **guarantee** these invariances by construction — not learned through data augmentation, but mathematically guaranteed.

---

### Slide 19: EGNN Architecture — Guaranteed Equivariance (1.5 min)

The **Equivariant Graph Neural Network**, or EGNN, achieves this guarantee through careful architectural design.

The key principle is: use only invariant quantities in the message computation. Distances between nodes are invariant — they don't change if we rotate or translate. So EGNN builds messages from distances, not raw coordinates.

But EGNN also updates node coordinates in an equivariant way. The coordinate update uses direction vectors — which are equivariant — scaled by learned weights. This allows the model to reason about geometry while maintaining mathematical guarantees.

The practical advantages are significant:
- No data augmentation needed for rotations — we get invariance for free
- Better generalization with less data — we don't waste capacity learning obvious symmetries
- Robustness is guaranteed by construction, not hopefully learned

---

### Slide 20: EGNN — Key Formulas (1 min)

Let me show the key formulas for EGNN.

For **message computation**: the message from node j to node i depends on their features and the squared distance between them. Note that the distance is invariant — it doesn't change under rotation.

For **coordinate update**: each node's position is updated by a weighted sum of direction vectors to its neighbors. The weights are learned and depend on the message. Crucially, direction vectors are equivariant, so the update preserves equivariance.

For **feature update**: node features are updated based on the aggregated messages, using a standard neural network.

This combination — invariant messages, equivariant coordinate updates — gives us the best of both worlds.

---

### Slide 21: Architecture Comparison (1 min)

Let me summarize the architectures we evaluated.

**GCN** is our simple baseline with normalized mean aggregation. It's effective but treats all neighbors equally.

**GAT** introduces attention to weight neighbor contributions adaptively. As we'll see, it achieves the best raw performance.

**DeepSets** uses global aggregation without explicit local structure. It's a useful comparison to assess the value of local structure.

**EGNN** guarantees E(3)-equivariance through its architecture. It offers a trade-off: slightly lower raw performance but guaranteed geometric robustness.

---

### Slide 22: Part 2 Summary (30 sec)

To summarize this theoretical section:

Graphs capture the relational structure of organoids with 1000× compression compared to raw images.

Graph Neural Networks learn from this non-Euclidean data through message passing.

GAT achieves the best performance through adaptive attention.

EGNN provides guaranteed geometric robustness through equivariant design.

---

## PART 3: METHODOLOGY (8-10 min)

### Slide 23: Part 3 Title (10 sec)

Let me now present our methodology: the complete pipeline from raw images to phenotype predictions.

---

### Slide 24: Pipeline Overview (1.5 min)

Here is an overview of our end-to-end pipeline.

We start with a 3D confocal image — about 2 gigabytes of data, 2048 by 2048 pixels times 200 slices.

The first step is **preprocessing**: intensity normalization and denoising to handle acquisition variations.

Next comes **cell segmentation** using Faster Cellpose — our optimized version of the state-of-the-art segmentation method. This produces masks labeling each individual cell.

We then extract **cell features**: centroid coordinates and volume, giving us 4 features per cell.

**DBSCAN clustering** separates individual organoids when multiple are present in the same field of view.

**Graph construction** via K-nearest-neighbors creates the geometric graph representation.

Finally, **GNN classification** predicts the phenotype.

The total processing time is about 20 minutes per organoid, dominated by segmentation. The output graph is only about 10 megabytes — a 1000× compression from the original image.

---

### Slide 25: Collaborative Dataset (1 min)

Our work relies on a substantial collaborative dataset collected through the ANR Morpheus project.

The data was acquired over 22 months, from May 2023 to February 2025, in collaboration with IPMC Nice and Paris Cité University.

This plot shows the cumulative growth of our dataset over time.

In total, we imaged 1,311 samples from which we extracted 2,272 individual organoids.

For this study, we selected 500 well-differentiated organoids — approximately 250 per class — where the phenotype label clearly matched the observed morphology. This ensures high-quality supervision for training.

---

### Slide 26: Segmentation Optimization (1.5 min)

Cell segmentation is a critical bottleneck. Let me explain our contribution here.

Cellpose is the current state-of-the-art for cell segmentation, achieving F1 scores of 0.98. But it's slow — 30 seconds per image slice.

For our dataset, this means: 30 seconds times 200 slices times 1000 organoids equals 2,500 hours of computation — over 100 days! This is prohibitive.

We developed **Faster Cellpose** through several optimizations.

First, **knowledge distillation**: we train a smaller "student" network to mimic the larger "teacher" Cellpose model, reducing parameters by 50%.

Second, **weight pruning**: we remove 30% of low-magnitude weights with minimal accuracy loss.

Third, **inference optimizations**: larger batch sizes, mixed precision computation, reduced iterations.

The result: 5× speedup while preserving F1 = 0.95. This reduces 2,500 hours to 500 hours — making our pipeline practical.

---

### Slide 27: Segmentation Methods Comparison (1 min)

This table compares our segmentation approaches.

Our geometric method based on ellipse detection is extremely fast — 3 seconds per slice — but achieves only 0.88 F1, which is too imprecise for our needs.

**Faster Cellpose**, our contribution, achieves 0.95 F1 at 6 seconds per slice — the optimal trade-off.

Original Cellpose achieves 0.98 F1 but at 30 seconds per slice — too slow for thousands of organoids.

We chose Faster Cellpose for our pipeline because segmentation quality directly impacts graph quality and downstream GNN performance.

---

### Slide 28: Geometric Graph Construction (1.5 min)

Once cells are segmented, we construct the geometric graph.

Each cell becomes a node with a **4-dimensional feature vector**: the 3D centroid coordinates (x, y, z) and the cell volume. Coordinates are normalized by centering and scaling.

For **edge construction**, we use K-nearest neighbors with K=10. Each node is connected to its 10 closest neighbors based on Euclidean distance between centroids.

We then **symmetrize** the graph: if i is a neighbor of j, then j is also a neighbor of i. We also apply a radial cutoff to remove excessively long edges.

This hybrid strategy balances connectivity, biological meaning, and computational efficiency.

The image on this slide shows graphs constructed from our data, clearly visualizing the cellular neighborhood structure.

---

### Slide 29: Synthetic Generation via Point Processes (1.5 min)

To address the scarcity of annotated data, we developed synthetic organoid generation based on spatial point processes.

The key insight is that the spatial distribution of cells differs between phenotypes.

For **cystic organoids**, cells are distributed relatively uniformly — we model this with a **homogeneous Poisson process**. This is complete spatial randomness: cells are positioned independently.

For **cauliflower organoids**, cells aggregate into clusters — we model this with a **Matérn cluster process**. Parent points are distributed uniformly, then child points cluster around each parent.

By controlling the process parameters, we generate a continuum from purely random to highly clustered distributions.

On this slide, you can see the visual difference: Poisson on the left with uniform distribution, Matérn on the right with clear clustering.

We generated **100,000 synthetic organoids** with perfectly known ground truth labels.

---

### Slide 30: From Synthetic to Real — Transfer Learning (1 min)

Our strategy is transfer learning from synthetic to real data.

**Phase 1: Pre-training** on 70,000 synthetic organoids. We train the GNN to regress the number of parent points in the Matérn process — a continuous measure of clustering degree. This teaches the network to recognize spatial organization patterns.

**Phase 2: Fine-tuning** on 500 real organoids. We reinitialize the classification head and fine-tune with a reduced learning rate.

The benefits are substantial:
- 4× data efficiency: 125 pre-trained organoids achieve what 500 from-scratch require
- 3× faster convergence
- 8% accuracy improvement

---

### Slide 31: Final Architecture (1 min)

Our final architecture uses GAT as the encoder — 5 layers with 256 hidden dimensions.

For global pooling, we concatenate mean and max pooling to capture both average and extreme features.

The classification head is a two-layer MLP: 256 to 128 to 2 outputs.

Training uses AdamW optimizer with learning rate 10⁻³ and dropout 0.15 for regularization.

For pre-training on synthetic data, we train for 200 epochs — about 48 hours on an NVIDIA RTX 3080.

For fine-tuning on real data, we use a reduced learning rate of 10⁻⁴ and train for 100 epochs — about 30 minutes.

---

### Slide 32: Part 3 Summary (30 sec)

To summarize our methodology:

We developed a complete automated pipeline from raw images to predictions.

Faster Cellpose provides 5× speedup while preserving segmentation quality.

Synthetic generation via point processes creates 100,000 training examples with perfect labels.

Transfer learning from synthetic to real reduces annotation needs by 4×.

---

## PART 4: EXPERIMENTAL RESULTS (10-12 min)

### Slide 33: Part 4 Title (10 sec)

Let me now present our experimental results, validating the choices I've described.

---

### Slide 34: GRETSI 2025 Comparative Study — Protocol (1 min)

Before showing results on real organoids, I want to present a fundamental study comparing GNNs to classical spatial statistics.

This work, published at GRETSI 2025, asks: when should we prefer GNNs over traditional methods like Ripley's K function?

Our protocol uses synthetic data where we control ground truth perfectly. We generate spherical point distributions — either Poisson (cystic) or Matérn (cauliflower) — and classify them.

We apply two types of noise to test robustness: Gaussian noise displacing point positions, and salt-and-pepper noise adding and removing points.

We compare GNN classifiers of varying depths to a Random Forest trained on Ripley's K, F, and G functions.

---

### Slide 35: Result 1 — Noise Robustness (1.5 min)

This figure shows accuracy versus Gaussian noise level.

The green curve is spatial statistics — Ripley's functions with Random Forest. The colored curves are GNNs of different depths.

The key observation: **spatial statistics are more robust to noise**. Even at high noise levels, they maintain over 85% accuracy. GNNs degrade more rapidly.

This makes sense: Ripley's functions average over many point pairs, naturally smoothing noise. GNNs propagate noise through message passing layers.

We also observe an optimal GNN depth around 5-6 layers. Deeper networks suffer from over-smoothing and overfit to noise.

---

### Slide 36: Result 2 — Geometric Generalization (1.5 min)

But here's the crucial finding that justifies our approach.

We trained both methods on perfectly spherical distributions. Then we tested on **ellipsoids** with increasing aspect ratios — from 1:1 (sphere) to 5:1 (elongated ellipsoid).

Look at this graph. Spatial statistics — the blue curve — collapse dramatically. From 100% accuracy on spheres to only 65% on ellipsoids. A 35% drop!

GNNs — the orange curve — degrade much more gracefully. From 95% to 82%. Only a 13% drop.

Why? Spatial statistics like Ripley's K depend explicitly on spherical geometry. Their theoretical values assume isotropic distances. When geometry deviates from spherical, the theory breaks down.

GNNs learn **topological patterns** that are more abstract: local clustering, cell arrangement, neighborhood structure. These patterns persist even when the overall shape changes.

For real organoids with variable morphologies, this geometric flexibility is decisive.

---

### Slide 37: Lesson from Comparative Study (1 min)

So what's the lesson?

**GNNs excel** when morphologies vary — exactly our situation with cystic versus cauliflower organoids having very different shapes.

**Spatial statistics excel** when geometry is perfectly spherical and consistent.

For real organoid analysis, where shapes range from smooth spheres to irregular cauliflowers, GNNs are the appropriate choice.

This rigorous comparison provides the foundation for our methodological choices.

---

### Slide 38: Performance on Synthetic Data (1.5 min)

Now let me show architecture comparison results on synthetic data.

The task is regression of the Matérn parent number — a continuous measure of clustering degree. We test on 15,000 synthetic organoids.

Here are the results in terms of Mean Squared Error.

GCN, our baseline, achieves 0.198.

DeepSets, without explicit local structure, achieves 0.145 — 27% better than GCN.

EGNN, with equivariance guarantees, achieves 0.137 — 31% better than GCN.

And GAT, with attention, achieves 0.118 — **40% better than GCN**.

GAT's attention mechanism allows it to adaptively weight which neighbors matter most for predicting clustering degree.

---

### Slide 39: Synthetic Results Analysis (1 min)

Let me interpret these results.

**GAT wins** because attention enables adaptive weighting. For detecting clustering patterns, not all neighbors are equally informative — attention learns this.

**DeepSets' good performance** is interesting. It means global statistics about cells — even without local structure — capture substantial information. But local structure still matters: GAT beats DeepSets significantly.

**EGNN's performance** represents a trade-off. It's slightly behind GAT in raw numbers, but it provides guaranteed equivariance — robustness to arbitrary rotations without data augmentation.

Our ablation studies show that equivariance divides the MSE by 2.8× compared to using raw coordinates naively. This is a major benefit.

---

### Slide 40: Performance on Real Data (1.5 min)

Now for the most important results: real organoid classification.

We evaluated on 500 well-differentiated organoids — approximately 250 cystic and 250 cauliflower — using 5-fold cross-validation.

Our main result: **84% accuracy** with GAT pre-trained on synthetic data.

Looking at per-class performance:
- **Cauliflower**: 93% precision, 74% recall. When we predict cauliflower, we're almost always right. But we miss some cauliflowers.
- **Cystic**: 78% precision, 95% recall. We catch almost all cystic organoids, with some false positives.

This asymmetry is interpretable: some cauliflower organoids have low deformation and appear quasi-spherical, leading to confusion with cystic.

---

### Slide 41: Confusion Matrix (1 min)

The confusion matrix shows the details.

On 75 test organoids:
- 28 cauliflower correctly classified
- 35 cystic correctly classified  
- 10 cauliflower misclassified as cystic
- 2 cystic misclassified as cauliflower

Total: 63 correct out of 75 — 84% accuracy.

The main confusion direction is cauliflower → cystic. These are often low-deformation cauliflower organoids with relatively smooth surfaces. Biologically, they may represent intermediate or transitional phenotypes.

---

### Slide 42: Transfer Learning Impact (1.5 min)

This table shows the crucial impact of transfer learning.

GAT trained from scratch on real data: 76% accuracy.

GAT pre-trained on synthetic, then fine-tuned on real: **84% accuracy**.

That's an **8 percentage point improvement** — substantial and consistent across cross-validation folds.

But even more importantly, look at **data efficiency**.

With only 25% of the real data — 125 organoids — the pre-trained model achieves 78% accuracy. That **matches** what the from-scratch model achieves with 100% of the data!

This means a **4× reduction in annotation requirements**. For practical applications where expert annotation is the bottleneck, this is transformative.

---

### Slide 43: Detailed Learning Curves (1 min)

These detailed learning curves show the pattern more clearly.

At 10% data — just 50 organoids — the pre-trained model gains 13 percentage points over from-scratch.

As we add more data, the gap narrows but remains: +11% at 25%, +10% at 50%, +8% at 100%.

The pre-trained model also converges 3× faster — 20-30 epochs versus 80-100 epochs.

This validates our synthetic generation strategy. Even though synthetic organoids differ from real ones, they teach useful representations of spatial organization that transfer effectively.

---

### Slide 44: Computational Efficiency (1 min)

Our pipeline is not only accurate but efficient.

For inference, we achieve throughput of over 200 organoids per minute in GPU batch mode. This enables high-throughput screening applications.

Memory footprint is modest — about 8 GB GPU memory — compatible with standard hardware.

Predictions are perfectly reproducible. Unlike human annotators with inter-observer variability, our model gives identical outputs every time.

For scalability: processing 1000 organoids takes about 17 hours with 20 GPUs running in parallel. Complete datasets can be analyzed in reasonable timeframes.

---

### Slide 45: Results Summary (1 min)

Let me summarize our experimental findings.

First, GNNs offer better geometric flexibility than spatial statistics — essential for variable organoid morphologies.

Second, GAT achieves the best performance: MSE of 0.118 on synthetic data, 84% accuracy on real data.

Third, transfer learning provides 4× data efficiency and 3× faster convergence.

Fourth, the pipeline is practical: 200+ organoids per minute, fully automated, perfectly reproducible.

---

### Slide 46: Identified Limitations (1 min)

I want to be transparent about limitations.

Our approach depends critically on segmentation quality. If segmentation fails, errors propagate through the entire pipeline.

We validated on prostate organoids only. Generalization to brain, liver, or other organoid types requires adaptation and validation.

We did not perform inter-laboratory validation. Robustness to different microscopes and protocols needs testing.

We did not include interpretability analysis — identifying which cells or regions drive predictions. This is important for biological understanding and acceptance.

---

## PART 5: CONCLUSION AND PERSPECTIVES (6-8 min)

### Slide 47: Part 5 Title (10 sec)

Let me now conclude with our contributions and perspectives for future work.

---

### Slide 48: Contribution 1 — Automated Pipeline (1 min)

Our first major contribution is a **complete automated pipeline** from raw images to predictions.

This pipeline handles the full workflow: preprocessing, segmentation, feature extraction, graph construction, and classification.

It achieves 1000× compression — from gigabytes of images to megabytes of graphs — while preserving biologically relevant information.

The entire codebase is open-source and available to the community.

---

### Slide 49: Contribution 2 — Segmentation Optimization (1 min)

Our second contribution addresses the segmentation bottleneck.

**Faster Cellpose** achieves 5× speedup through knowledge distillation and pruning, while maintaining F1 = 0.95.

We also developed a geometric method based on ellipse detection achieving 15× speedup, useful for ultra-rapid primary screening.

These optimizations make high-throughput analysis practical — thousands of organoids instead of hundreds.

---

### Slide 50: Contribution 3 — Geometric Graphs and GNNs (1 min)

Our third contribution is the graph-based representation and GNN analysis framework.

We showed that geometric graphs explicitly capture cellular relational structure — which cells are neighbors, how they're organized spatially.

We systematically compared four architectures: GAT, DeepSets, EGNN, and GCN.

We demonstrated the value of E(3)-equivariant architectures for guaranteed geometric robustness.

---

### Slide 51: Contribution 4 — Synthetic Generation (1 min)

Our fourth contribution is synthetic data generation via spatial point processes.

Using Poisson and Matérn processes, we generated 100,000 synthetic organoids with perfectly known labels.

This enabled pre-training that improves real-data performance by 8% and reduces annotation needs by 4×.

The methodology is generalizable to other spherical or ellipsoidal biological structures.

---

### Slide 52: Contribution 5 — GRETSI Comparative Study (45 sec)

Finally, our rigorous comparative study, published at GRETSI 2025, provides foundational understanding.

We showed that GNNs offer better geometric generalization — essential for variable morphologies.

Spatial statistics offer better noise robustness — valuable when geometry is consistent.

This guides method selection based on data characteristics.

---

### Slide 53: Short-Term Perspectives (1.5 min)

Looking ahead, several extensions are immediately feasible.

For **methodological extensions**, we plan to develop multi-scale graphs capturing cell, region, and organoid levels simultaneously. Graph Transformer architectures could provide global attention. Alpha-shape morphological descriptors could complement local cellular analysis.

For **multi-modal integration**, spatial transcriptomics data could be combined with imaging — each cell gets both spatial and molecular features. Temporal data from time-lapse imaging could capture developmental dynamics.

For **clinical validation**, prospective studies on patient cohorts would test therapeutic response prediction.

---

### Slide 54: Long-Term Perspectives (1 min)

The long-term vision is transformative.

**Therapeutic response prediction**: Patient-derived organoids could be tested against multiple treatments to predict which therapy will work best for that individual patient.

**Generative graph models**: In silico organoid design could optimize culture protocols computationally before wet-lab experiments.

**Societal impact**: Organoid-based screening follows the 3Rs principles — Replace, Reduce, Refine animal experimentation. Our automated analysis tools accelerate this transition.

---

### Slide 55: Final Message (1 min)

Let me conclude with a final message.

**Geometric Graph Neural Networks** are a powerful tool for 3D organoid analysis. They capture relational structure, provide geometric robustness, and achieve high performance.

There is a **virtuous synergy between biology and AI**: better data enables better models, which generate better understanding, which informs better experiments.

Our **open-source code** is available for the community to use, adapt, and extend.

I believe this work contributes to an exciting future where AI-powered organoid analysis accelerates biomedical discovery and improves human health.

Thank you for your attention. I am happy to take your questions.

---

## NOTES FOR DELIVERY

### General Tips
- Speak clearly and at moderate pace (~130 words/min)
- Make eye contact with jury members
- Use the laser pointer sparingly and purposefully
- Pause briefly between sections
- Stay calm if asked difficult questions

### Key Points to Emphasize
- The 1000× compression while preserving biological information
- GAT as best performer, EGNN for guaranteed robustness
- 4× data efficiency from transfer learning
- 84% accuracy on real organoids
- Practical applicability: 200+ organoids/minute

### Anticipated Questions
1. **Why GAT over EGNN?** Trade-off between raw performance and guaranteed robustness. GAT wins on accuracy; EGNN wins on theoretical guarantees.

2. **Synthetic data validation?** Acknowledged limitation — no formal statistical validation. But empirical gains (8%) demonstrate practical utility.

3. **Generalization to other organoids?** Principles are transferable; fine-tuning Cellpose and adapting features would be needed.

4. **Why only 500 organoids selected?** Quality over quantity — ensuring label-morphology consistency for reliable supervised learning.

5. **Clinical deployment?** Would require regulatory validation (IVDR/FDA), multi-site testing, user interface development.

### Timing Checkpoints
- After Introduction (Slide 8): ~7 min
- After Foundations (Slide 22): ~20 min
- After Methodology (Slide 32): ~29 min
- After Results (Slide 46): ~40 min
- End: ~47 min

If running long: abbreviate slides 38-39 (synthetic results details)
If running short: expand on perspectives (slides 53-54)

