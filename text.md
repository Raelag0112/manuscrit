# Defense Speech — English Speaking Text

**Thesis**: Deep Learning for Organoid Analysis: Graph-Based Modeling of 3D Cellular Architectures  
**Candidate**: Alexandre Martin  
**Duration**: 45 minutes

---

## INTRODUCTION (5-6 min)

### Slide 1: Title (30 sec)

Good afternoon everyone, and thank you for being here today.

I am Alexandre Martin, and I am honored to present my doctoral thesis entitled "Deep Learning for Organoid Analysis: Graph-Based Modeling of 3D Cellular Architectures."

This work was conducted at INRIA Sophia-Antipolis within the Morpheme team, in collaboration with IPMC Nice and Paris Cité University, as part of the ANR Morpheus project.

---

### Slide 2: Organoids Introduction (1 min)

Let me start by introducing organoids. Organoids are often called "mini-organs in a dish." They are three-dimensional structures grown in vitro from stem cells that self-organize to mimic the architecture and function of real organs.

Unlike traditional two-dimensional cell cultures, organoids reproduce the spatial complexity we observe in vivo. They bridge the gap between oversimplified 2D cultures and expensive, ethically problematic animal models.

Their applications are transformative: personalized medicine, drug screening, and disease modeling.

---

### Slide 3: The Challenge (1 min)

In this thesis, we focus on prostate organoids. We observe two main phenotypes that are our classification targets.

The **cystic phenotype** represents healthy organoids: spherical shape, central cavity, organized structure, and uniform cell distribution.

The **cauliflower phenotype** represents a perturbed state: irregular surface, multiple buds, and clustered cell distribution.

Our goal is to automatically classify these phenotypes from 3D microscopy images.

---

### Slide 4: Starting Point — No Data (1 min)

Here is a crucial point that shaped this entire thesis: **at the beginning, we had no annotated data**.

Unlike computer vision where ImageNet provides 14 million labeled images, the organoid domain had no public datasets. Expert annotation is expensive — a biologist's time costs around 50 euros per hour. Getting thousands of annotations is simply not practical.

This posed a fundamental question: **how to develop and validate methods without real data?**

Our answer was to bootstrap the research using **synthetic data** combined with **graph modeling**. This is the story I will tell you today.

---

### Slide 5: Three Key Challenges (1 min)

Let me crystallize these into three key challenges.

**Challenge 1 — Manual Analysis**: Currently, biologists analyze organoids manually. This takes 30 minutes per organoid, is subjective, and not reproducible. For drug screening with thousands of organoids, this simply doesn't scale.

**Challenge 2 — Data Scarcity**: Unlike ImageNet with 14 million images, we have no public organoid datasets. Expert annotation is expensive — 50 euros per hour. Getting thousands of annotations is impractical.

**Challenge 3 — 3D Geometry**: Organoids float freely in 3D culture with no preferred orientation. Each organoid is about 2 gigabytes of data. We need methods that are rotation-invariant and memory-efficient.

We need automated, data-efficient, and geometrically robust methods.

---

### Slide 6: Research Questions (45 sec)

These constraints lead us to three core research questions.

**Question 1**: Can we model organoids as geometric graphs? Can we capture the essential biological information while compressing the data?

**Question 2**: Do GNNs outperform classical spatial statistics? We needed to justify the deep learning approach rigorously.

**Question 3**: Can synthetic data enable transfer learning? Can we compensate for the lack of real annotations?

---

### Slide 7: Presentation Outline (30 sec)

Here is my presentation outline, which reflects the actual research journey.

**Part I**: I'll start with graph modeling and justify why we chose GNNs — this was our work when we had no real data.

**Part II**: Then I'll present the theoretical foundations of Graph Neural Networks.

**Part III**: When real data finally arrived, we built a processing pipeline — I'll describe this next.

**Part IV**: Finally, I'll show the transfer learning results combining synthetic and real data.

The narrative follows our research path: no data → synthetic + theory → real data arrives → transfer learning.

---

## PART I: GRAPH MODELING & JUSTIFICATION (12 min)

### Slide 8: Part I Title (10 sec)

Let me now explain why we chose to model organoids as graphs and why we chose GNNs over classical methods.

---

### Slide 9: Why Model Organoids as Graphs? (1.5 min)

The key biological insight is that **cell spatial organization** differs between phenotypes. Cystic organoids have uniformly distributed cells; cauliflower organoids have clustered cells.

This leads naturally to a graph abstraction: each **cell becomes a node**, spatial **proximity defines edges**, and node features capture position and volume.

This representation offers spectacular advantages:
- **1000× compression**: from gigabytes of images to megabytes of graphs
- It captures the relational structure — which cells are neighbors
- It's natural for point cloud data

On this slide, you can see the transformation from 3D image to point cloud to geometric graph.

---

### Slide 9: Synthetic Organoid Generation — Motivation (1 min)

With no real data at the start of this thesis, we needed to create synthetic organoids.

The key biological insight is that **cell distribution differs between phenotypes**: cystic organoids have uniformly distributed cells, while cauliflower organoids have clustered cells.

Our solution uses **spatial point processes** — mathematical models for generating point patterns with controllable properties. This gives us perfect ground truth labels and unlimited training data.

---

### Slide 10: Point Processes — Mathematical Foundation (1.5 min)

Let me explain the mathematical foundation.

For **cystic organoids**, we use a **homogeneous Poisson process**. Points are distributed with complete spatial randomness — no clustering, no regularity. The intensity parameter λ controls the expected number of points per unit volume.

For **cauliflower organoids**, we use a **Matérn cluster process**. First, we generate parent points uniformly. Then, around each parent, we scatter children points. The parameters control parent intensity κ, children per parent μ, and cluster radius r.

By varying these parameters, we generate a continuum from uniform to highly clustered distributions.

---

### Slide 11: Synthetic Dataset Statistics (1 min)

Our synthetic dataset is substantial.

For generation parameters, each organoid contains 100 to 500 cells, distributed on a normalized unit sphere. Cell volumes follow a log-normal distribution matching real biology.

We generated **100,000 synthetic organoids** total: 70,000 for training, 15,000 for validation, and 15,000 for testing.

The key advantages are **perfect ground truth** — we know exactly how clustered each organoid is — **unlimited supply** — we can generate more anytime — and **controllable difficulty** — we can create easy or hard cases.

---

### Slide 13: GRETSI Study — GNNs vs Spatial Statistics (1 min)

Before committing to GNNs, we asked: **when should we prefer GNNs over classical methods?**

This rigorous comparison, published at GRETSI 2025, compares:
- **Classical approach**: Ripley's K, F, G functions with Random Forest
- **Deep learning approach**: Graph Neural Networks

Our protocol uses synthetic data where we control ground truth. We test noise robustness and geometric generalization.

---

### Slide 14: Result 1 — Noise Robustness (1.5 min)

This figure shows accuracy versus noise level.

The key observation: **spatial statistics are more robust to noise**. Even at high noise levels, Ripley's functions maintain good accuracy.

Why? Ripley's functions average over many point pairs, naturally smoothing noise. GNNs can propagate noise through message passing layers.

We also observe an optimal GNN depth around 5-6 layers — deeper networks suffer from over-smoothing.

---

### Slide 15: Result 2 — Geometric Generalization (1.5 min)

But here's the crucial finding that justifies our approach.

We trained both methods on perfect spheres. Then we tested on **ellipsoids** with increasing aspect ratios.

Look at this graph:
- **Spatial statistics** collapse from 100% to 65% — a 35% drop!
- **GNNs** degrade gracefully from 95% to 82% — only 13% drop.

Why? Ripley's K assumes spherical geometry. When shapes deviate, the theory breaks.

GNNs learn **topological patterns** — local clustering, neighborhood structure — that persist even when overall shape changes.

For real organoids with variable morphologies, this geometric flexibility is decisive.

---

### Slide 16: Justification — Why GNNs for Organoids (1 min)

So what's the lesson?

**GNNs excel** when morphologies vary, shapes are irregular, geometry is non-spherical — exactly our situation with real organoids.

**Spatial statistics excel** when geometry is perfectly spherical and consistent — idealized conditions only.

This rigorous comparison provides the foundation for choosing GNNs.

---

### Slide 17: Part I Summary (30 sec)

To summarize Part I:

Graphs naturally capture cellular spatial organization with 1000× compression.

Synthetic data enabled method development without annotations.

The GRETSI study demonstrates GNNs outperform spatial statistics for variable morphologies — justifying our deep learning approach.

Now that we've justified GNNs, let's understand how they work.

---

## PART II: GNN THEORY (12 min)

### Slide 18: Part II Title (10 sec)

Let me now present the theoretical foundations of Graph Neural Networks.

---

### Slide 19: Why Not 3D CNNs? (1 min)

A natural question: why not use 3D CNNs?

**Memory constraints**: A single organoid is about 2 GB — 2048×2048×200 voxels. GPU memory is prohibitive.

**Downsampling destroys details**: Reducing resolution loses individual cells.

**No native invariance**: 3D CNNs aren't invariant to rotations. Augmentation is expensive and doesn't provide guarantees.

**Grid limitation**: CNNs treat organoids as voxel grids, not as cell networks.

Graphs and GNNs overcome all these limitations.

---

### Slide 20: Message Passing Paradigm (1.5 min)

GNNs solve the challenge of learning from graphs through **message passing**.

The idea is elegant: each node iteratively aggregates information from its neighbors.

The update rule: node i's new representation equals an UPDATE function applied to its current representation and an AGGREGATE of neighbors' representations.

After L layers, each node captures information from its L-hop neighborhood.

This is analogous to convolution on images, generalized to arbitrary graph structures.

---

### Slide 18: Global Pooling (1 min)

For classification, we need to go from **node-level** representations to a **graph-level** representation.

We have several pooling options: **Mean pooling** averages all node features, **Max pooling** takes element-wise maximum, **Sum pooling** adds all features, and **Attention pooling** uses learned weights.

Our choice is **Mean + Max concatenation** — combining both gives complementary information: mean captures overall statistics, max captures most salient features.

The full pipeline is: nodes pass through the GNN to get embeddings, then we pool to get a single graph vector, then an MLP produces the class prediction.

---

### Slide 22: GCN — Baseline (1 min)

The **Graph Convolutional Network** is our baseline, introduced by Kipf and Welling in 2017.

It performs normalized mean aggregation: each node's update is a weighted average of neighbors' features.

GCN is simple and effective, but it treats all neighbors equally — no distance or importance weighting.

---

### Slide 23: GAT — Our Main Choice (1.5 min)

The **Graph Attention Network** addresses this through learned attention.

The key idea: learn attention coefficients that weight neighbor importance. Not all neighbors contribute equally — some are more informative.

We compute attention scores for each neighbor pair, apply softmax normalization, then aggregate weighted features.

GAT also uses **multi-head attention** — multiple independent mechanisms capturing different relationship types.

As we'll see, **GAT achieves the best performance** precisely because of this adaptive weighting.

---

### Slide 24: DeepSets — Comparison (1 min)

As a baseline, we also evaluated **DeepSets** — a set-based approach.

DeepSets treats the organoid as an unordered set of cells, ignoring neighborhood structure. Each cell is encoded independently, then globally aggregated.

Surprisingly, it performs reasonably well — global statistics are informative. But architectures with explicit local structure perform better.

---

### Slide 25: Geometric Equivariance (1 min)

Now a crucial concept: **geometric equivariance**.

Organoids in 3D culture have no preferred orientation. The biological phenotype doesn't depend on how we orient the sample.

**Invariance** means the prediction is identical regardless of rotation: f(Rx) = f(x).

**Equivariance** means intermediate representations rotate coherently: f(Rx) = R·f(x).

We can build architectures that **guarantee** these properties mathematically.

---

### Slide 26: The E(3) Group (1 min)

Before diving into EGNN, let me explain the E(3) group — the Euclidean group in 3 dimensions.

E(3) includes three types of transformations: **Rotations** — any 3D rotation around any axis, **Translations** — shifting the entire structure in space, and **Reflections** — mirror symmetries.

The key insight is that we can build architectures that **guarantee** these invariances by construction — not learned through augmentation, but **mathematically proven**. This is much stronger than hoping data augmentation will teach the network rotation invariance.

---

### Slide 27: EGNN — Guaranteed Equivariance (1.5 min)

The **Equivariant Graph Neural Network** achieves this guarantee.

The key principle: use only **invariant quantities** in messages. Distances don't change under rotation — so EGNN builds messages from distances, not raw coordinates.

Coordinate updates use direction vectors — equivariant quantities — scaled by learned weights.

Practical advantages:
- No data augmentation needed
- Better generalization with less data
- Robustness guaranteed by construction

The trade-off: slightly lower raw accuracy than GAT.

---

### Slide 28: Architecture Comparison (1 min)

Let me summarize:

**GCN**: Simple baseline, treats neighbors equally.

**GAT**: Attention-weighted aggregation, **best accuracy**.

**DeepSets**: Global aggregation, no local structure.

**EGNN**: Distance-based, **guaranteed E(3)-equivariance**.

The key trade-off: GAT for best raw performance, EGNN for guaranteed geometric robustness.

---

### Slide 29: Part II Summary (30 sec)

To summarize Part II:

Message passing enables learning on graphs.

GAT with attention achieves best accuracy.

EGNN provides guaranteed E(3)-equivariance.

All architectures outperform spatial statistics on variable morphologies.

Next: real data finally arrives — how to build the processing pipeline?

---

## PART III: REAL DATA PIPELINE (10 min)

### Slide 30: Part III Title (10 sec)

Let me now describe how we transform real 3D images into graphs.

---

### Slide 31: The Real Dataset Arrives (1.5 min)

Through the ANR Morpheus collaboration, real data finally arrived.

Over 22 months — May 2023 to February 2025 — IPMC Nice and Paris Cité collected 1,311 samples, from which we extracted 2,272 organoids.

For this study, we selected **500 well-differentiated organoids** — approximately 250 per class — where labels clearly match morphology.

The challenge: how to transform 2 GB 3D images into graphs for GNN processing?

---

### Slide 27: End-to-End Pipeline (1.5 min)

Here is our complete pipeline.

We start with a 3D confocal image — 2 GB of data.

**Preprocessing** handles intensity normalization and denoising.

**Faster Cellpose** — our optimized segmentation — identifies individual cells.

**Feature extraction** computes centroids and volumes.

**DBSCAN clustering** separates organoids in multi-organoid fields.

**Graph construction** via K-nearest-neighbors creates the geometric graph.

**GNN classification** predicts the phenotype.

Total time: ~20 minutes per organoid. Compression: 2 GB → 10 MB — **1000× reduction**.

---

### Slide 28: Faster Cellpose (1.5 min)

Cell segmentation is the bottleneck. Cellpose is state-of-the-art but slow — 30 seconds per slice.

For our dataset: 30 sec × 200 slices × 1000 organoids = **2,500 hours**. Prohibitive!

We developed **Faster Cellpose** through:
- **Knowledge distillation**: smaller student network, −50% parameters
- **Weight pruning**: remove 30% low-magnitude weights
- **Mixed precision**: FP16 inference

Result: **5× speedup** while maintaining F1 = 0.95. This makes the pipeline practical.

---

### Slide 29: Graph Construction (1 min)

From segmented cells, we construct graphs.

**Node features** (4D): 3D centroid coordinates plus cell volume, normalized.

**Edges**: K-nearest neighbors with K=10, Euclidean distance, symmetrized with radial cutoff.

This captures the cellular neighborhood structure essential for phenotype classification.

---

### Slide 30: Part III Summary (30 sec)

To summarize Part III:

Real dataset: 500 well-differentiated organoids collected over 22 months.

Faster Cellpose: 5× speedup maintaining quality.

Graph construction: 1000× compression preserving structure.

Complete pipeline: 3D image → graph → prediction.

Next: can transfer learning from synthetic data improve real-world performance?

---

## PART IV: RESULTS & TRANSFER LEARNING (10 min)

### Slide 31: Part IV Title (10 sec)

Let me now present the transfer learning results.

---

### Slide 32: Transfer Learning Strategy (1.5 min)

Our strategy bridges synthetic and real data.

**Phase 1: Pre-training** on 70,000 synthetic organoids. We train the GNN to regress the Matérn parent number — a continuous clustering measure. This teaches spatial organization patterns.

**Phase 2: Fine-tuning** on 500 real organoids. We reinitialize the classification head and fine-tune with reduced learning rate.

The question: does synthetic pre-training help with real classification?

---

### Slide 33: Performance on Synthetic Data (1.5 min)

First, architecture comparison on synthetic data.

Task: regression of Matérn parent number. Test set: 15,000 organoids.

Results (Mean Squared Error):
- GCN (baseline): 0.198
- DeepSets: 0.145 (+27%)
- EGNN: 0.137 (+31%)
- **GAT: 0.118 (+40%)**

GAT wins because attention learns which neighbors matter for detecting clustering patterns.

EGNN's equivariance divides MSE by **2.8×** compared to naive coordinate use.

---

### Slide 34: Synthetic Results Analysis (1 min)

Let me analyze these synthetic results more deeply.

**Why does GAT win?** Attention learns **which neighbors matter** — it's adaptive to clustering patterns, and multi-head attention captures different relationship types simultaneously.

**Why does DeepSets perform well?** Global statistics are informative for distinguishing clustering patterns. But local structure still helps, which is why GAT beats DeepSets.

The **EGNN trade-off** is interesting: it's slightly behind GAT in raw MSE, but it provides guaranteed equivariance. Without equivariance, MSE was 0.38. With EGNN, it's 0.137 — a **2.8× improvement** from the geometric guarantee alone.

---

### Slide 35: Performance on Real Data (1.5 min)

Now the crucial result: real organoid classification.

500 organoids, 5-fold cross-validation.

**84% accuracy** with GAT pre-trained on synthetic data.

Per-class performance:
- **Cauliflower**: 93% precision, 74% recall
- **Cystic**: 78% precision, 95% recall

The main confusion: low-deformation cauliflowers appear quasi-spherical, resembling cystic organoids.

---

### Slide 36: Confusion Matrix Analysis (1 min)

Let me show the confusion matrix in detail.

We have 28 true positive cauliflowers, 35 true positive cystics, and 63 out of 75 correct predictions — that's our 84%.

The **main error pattern** is cauliflower misclassified as cystic — 10 cases. Only 2 cystics were misclassified as cauliflower.

The **interpretation**: low-deformation cauliflowers have quasi-spherical appearance. They may represent transitional phenotypes — biologically intermediate states. This is actually informative about the biology!

---

### Slide 37: Transfer Learning Impact (1.5 min)

This slide shows why transfer learning matters.

GAT from scratch: 76% accuracy.
GAT pre-trained: **84% accuracy**.
**+8 percentage points improvement**.

But even more important — **data efficiency**:
- Pre-trained with 25% data (125 organoids): 78%
- From-scratch with 100% data: 76%

The pre-trained model with 25% of data **matches** the from-scratch model with 100%!

This means **4× reduction in annotation requirements** — transformative for practical applications.

Convergence is also **3× faster**.

---

### Slide 36: Final Architecture (1 min)

Our final architecture:

**Encoder**: GAT, 5 layers, 256 hidden dimensions.

**Pooling**: Mean + Max concatenation.

**Head**: MLP 256 → 128 → 2 classes.

**Training**: AdamW optimizer, 10⁻³ pre-train / 10⁻⁴ fine-tune.

**Inference**: 200+ organoids/minute, ~8 GB GPU memory.

---

### Slide 39: Learning Curves — Detailed Analysis (1 min)

Let me show the learning curves in detail.

The data efficiency gains are striking. With 10% data — just 50 organoids — from-scratch gets 58%, but pre-trained gets 71% — that's **+13%**. At 25% data, pre-trained reaches 78%, which matches from-scratch at 100%. At full data, we see the +8% improvement.

The **key observation**: largest gains happen in the **low data regime**. This is exactly where we need help most!

For convergence, pre-trained models converge in 20-30 epochs, while from-scratch needs 80-100 epochs — **3× faster**.

The practical impact is transformative: 125 real organoids with pre-training performs equivalently to 500 organoids from scratch.

---

### Slide 40: Computational Efficiency (1 min)

Let me discuss computational efficiency, crucial for practical deployment.

**Inference performance**: We achieve **200+ organoids per minute** using GPU batch processing. This enables high-throughput screening applications.

**Memory footprint**: About 8 GB GPU memory, compatible with standard hardware like RTX 3080 or A100.

**Reproducibility**: Results are **100% deterministic**. No inter-observer variability. Identical results every run — essential for scientific validity.

**Scalability**: 100 organoids in 30 seconds, 1000 in 5 minutes, 10,000 in 50 minutes.

Compare to manual analysis: 30 minutes per organoid times 1000 equals 500 hours. Our automated system: 5 minutes total. That's **6000× faster**.

---

### Slide 41: Limitations (1 min)

I want to be transparent about limitations.

**Segmentation dependency**: Errors propagate through the pipeline.

**Validation scope**: Prostate organoids only; generalization needs testing.

**No inter-laboratory validation**: Different microscopes and protocols.

**No interpretability**: Which cells drive predictions?

---

### Slide 42: Part IV Summary (30 sec)

To summarize results:

GAT achieves 84% accuracy on real data.

Transfer learning provides +8% improvement, 4× data efficiency, 3× faster convergence.

Practical system: 200+ organoids/minute, fully automated.

---

## CONCLUSION (5 min)

### Slide 39: Conclusion Title (10 sec)

Let me conclude with our contributions and perspectives.

---

### Slide 44: Contribution 1 — Graph-Based Representation (1 min)

My first contribution is the **graph-based representation** for organoid modeling.

The key innovation is the cell-to-node abstraction: each cell becomes a node, spatial proximity defines edges, and we use 4D features capturing position and volume.

The impact is spectacular: **1000× compression** — from 2 GB images to 2 MB graphs — while preserving the biological structure that matters for classification. This enables GNN analysis that would be impossible on raw images.

---

### Slide 45: Contribution 2 — GRETSI Comparative Study (1 min)

My second contribution is the **GRETSI comparative study**, published at GRETSI 2025.

We rigorously compared GNNs versus spatial statistics. The key findings: GNNs provide better **geometric generalization** to varying shapes, while spatial statistics offer better **noise robustness**.

The recommendation: use GNNs for variable morphologies like real organoids, statistics only for idealized spherical conditions. This justifies our entire deep learning approach and provides guidelines for the community.

---

### Slide 46: Contribution 3 — Synthetic Data Generation (1 min)

My third contribution is the **synthetic data generation** system using point processes.

We use Poisson processes for uniform distributions (cystic) and Matérn cluster processes for clustered distributions (cauliflower). We generated **100,000 synthetic organoids** with perfect ground truth labels.

This enabled research when we had no real data, provided the foundation for transfer learning, and delivered +8% accuracy improvement on real data. The approach is generalizable to other spherical or ellipsoidal biological structures.

---

### Slide 47: Contribution 4 — Transfer Learning Strategy (1 min)

My fourth contribution is the **transfer learning strategy** from synthetic to real data.

The results speak for themselves: **+8%** accuracy improvement, **4×** data efficiency, and **3×** faster convergence.

The practical impact is transformative: 125 organoids with pre-training equals 500 organoids from scratch. We fundamentally changed the annotation requirements for organoid analysis.

---

### Slide 48: Contribution 5 — Complete Automated Pipeline (1 min)

My fifth contribution is the **complete automated pipeline** — end-to-end from raw 3D images to phenotype predictions.

Components include preprocessing, Faster Cellpose with 5× speedup, graph construction, and GNN classification. The system is **fully automated**, runs at **200+ organoids per minute**, and is completely **open-source**.

The code is available at github.com/morpheme-inria/organoid-gnn for the community to use and build upon.

---

### Slide 49: Short-Term Perspectives (1 min)

Looking at short-term perspectives.

**Methodological extensions** include multi-scale graphs — from cell to region to organoid levels — Graph Transformers with global attention, and interpretability analysis to understand which cells drive predictions.

**Multi-modal integration** would combine spatial transcriptomics with imaging, temporal data from time-lapse microscopy, and metabolomic features.

**Clinical validation** requires testing on prospective patient cohorts and multi-site validation.

---

### Slide 50: Long-Term Vision (1 min)

For the long-term vision.

**Therapeutic response prediction** using patient-derived organoids: test treatments in silico and enable personalized therapy selection.

**Generative graph models** for in silico organoid design, optimizing culture protocols through computational biology.

**Multi-organ extension** to brain organoids, liver, kidney, lung, and tumor organoids.

The **societal impact** supports the 3Rs principles — Replace, Reduce, Refine animal experimentation — while accelerating drug discovery.

---

### Slide 52: Final Message (1 min)

Let me conclude with a final message.

**Geometric Graph Neural Networks** are a powerful tool for 3D organoid analysis.

Our research journey: starting with no data → developing synthetic approaches → justifying GNNs → building pipelines when real data arrived → demonstrating transfer learning success.

Our **open-source code** is available for the community.

I believe this work contributes to an exciting future where AI-powered organoid analysis accelerates biomedical discovery.

Thank you for your attention. I am happy to take your questions.

---

## NOTES FOR DELIVERY

### Key Points to Emphasize
- The narrative: no data → synthetic + theory → real data → transfer learning
- 1000× compression while preserving biological information
- GRETSI justification for GNNs over spatial statistics
- GAT as best performer, EGNN for guaranteed robustness
- 4× data efficiency from transfer learning
- 84% accuracy on real organoids
- Practical applicability: 200+ organoids/minute
- 6000× faster than manual analysis

### Anticipated Questions
1. **Why GAT over EGNN?** Trade-off: GAT wins on accuracy; EGNN wins on theoretical guarantees.

2. **Synthetic data validation?** Empirical gains (+8%) demonstrate practical utility despite domain gap.

3. **Generalization to other organoids?** Principles transferable; requires fine-tuning Cellpose and adapting features.

4. **Why only 500 organoids?** Quality over quantity — ensuring label-morphology consistency.

5. **Clinical deployment?** Would require regulatory validation (IVDR/FDA), multi-site testing.

6. **Why point processes specifically?** They naturally model the biological phenomena: Poisson for random cell distribution, Matérn for clustering.

### Timing Checkpoints
- After Introduction (Slide 7): ~6 min
- After Part I - Justification (Slide 17): ~16 min
- After Part II - GNN Theory (Slide 28): ~26 min
- After Part III - Pipeline (Slide 35): ~33 min
- After Part IV - Results (Slide 43): ~43 min
- Conclusion (Slide 52): ~45 min

If running long: abbreviate E(3) group details (Slide 22) or Contribution slides (44-48)
If running short: expand on perspectives and limitations

### General Tips
- Emphasize the research narrative: how constraints shaped the approach
- Speak clearly and at moderate pace (~130 words/min)
- Make eye contact with jury members
- Pause briefly between sections
- Stay calm if asked difficult questions
- Use the confusion matrix to discuss biological insights
