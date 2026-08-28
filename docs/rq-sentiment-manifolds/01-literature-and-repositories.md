# Literature and repository synthesis

## What was verified

The supplied report was treated as source material, not as instructions. Its four-paper framing is productive, but several details needed correction against the papers themselves. In particular, Tigges et al. use decoder-only GPT-2/Pythia models and a specific toy/SST/OpenWebText program; the later geometry papers do not already establish that sentiment manifolds are causally superior to a direction.

## The four papers

### 1. Tigges et al.: the baseline to beat

**Paper:** Curt Tigges, Oskar J. Hollinsworth, Atticus Geiger, and Neel Nanda, [*Language Models Linearly Represent Sentiment*](https://aclanthology.org/2024.blackboxnlp-1.5/) (BlackboxNLP 2024). The earlier arXiv title is [*Linear Representations of Sentiment in Large Language Models*](https://arxiv.org/abs/2310.15154).

**Official code:** [curt-tigges/eliciting-latent-sentiment](https://github.com/curt-tigges/eliciting-latent-sentiment).

**Core experimental program.** The paper studies GPT-2 and Pythia models from 85M to 2.8B. It learns sentiment directions from residual-stream activations using mean differences, K-means centroid differences, logistic-regression weights, PCA, and Distributed Alignment Search (DAS). It then uses directional activation patching, activation addition, and directional ablation to ask whether the direction is used by the model.

**Datasets and tasks.**

- **ToyMovieReview:** prompts such as `I thought this movie was ADJECTIVE, I VERBed it. Conclusion: This movie is`, with 85 adjectives split 55/30 and eight held-out/controlled verbs.
- **ToyMoodStory:** templated multi-subject stories, including a completion contrast such as `excited` vs. `nervous`.
- **Stanford Sentiment Treebank (SST):** binary-collapsed real movie-review phrases for OOD causal validation.
- **OpenWebText:** naturalistic text used to inspect correlational behavior of the learned direction.

**The hard numerical baseline.** On Pythia-1.4B, selecting the best layer for each method, DAS obtains about 109.8% ToyMovieReview logit-difference recovery, 100% toy logit flips, 47.0% SST logit-difference recovery, and 53.5% SST logit flips. Mean difference, K-means, logistic regression, and PCA are weaker. Crucially, 2D and 3D DAS slightly improve or match in-sample recovery but generalize *worse* to SST than 1D DAS. This is an existing capacity-control warning: simply increasing representation dimension is not enough.

**Summarization motif.** In Pythia-2.8B zero-shot SST classification, ablating the sentiment direction across all tokens/layers reduces accuracy from 100% to 62%. Ablating it only at comma positions reduces accuracy to 82%. The direction is therefore written not just at sentiment-bearing words but at apparently neutral summary positions such as punctuation and names.

**What a manifold paper should preserve.**

- The same clean/corrupted pair construction and directional patching semantics.
- Toy-to-SST generalization, rather than merely training-set decoding.
- Exact layer and token-position sweeps.
- Random-direction and full-activation controls.
- Mechanistic localization, including the summarization motif.

**Openings left by the paper.** Tigges et al. explicitly allow that a direction may be the first PC of more task-specific directions, that a feature can split across contexts, that positive/negative and valenced/unvalenced may differ, and that causal abstractions do not explain all behavior. They also flag small datasets, hyperparameter sensitivity, OOD activation addition, and limited MLP analysis. Those are legitimate entry points for a manifold account.

### 2. Lyngbæk et al.: the neutral residual and portability

**Paper:** Laurits Lyngbæk, Pascale Feldkamp, Yuri Bizzoni, Kristoffer L. Nielbo, and Kenneth Enevoldsen, [*Is Sentiment Banana-Shaped? Exploring the Geometry and Portability of Sentiment Concept Vectors*](https://aclanthology.org/2026.wassa-1.13/) (WASSA 2026); [arXiv](https://arxiv.org/abs/2601.07995).

**Official code:** [lauritswl/representation-transfer](https://github.com/lauritswl/representation-transfer).

**Core contribution.** Concept Vector Projection (CVP) uses the normalized difference between positive and negative sentence-embedding means. With multilingual MPNet sentence embeddings, the authors test transfer among EmoBank (10,062 English sentences), Facebook status updates (2,895), and Fiction4 (6,300 Danish literary excerpts). Cross-corpus Spearman correlations are often around 0.64–0.70, showing substantial portability.

**Geometric clue.** The authors compare negative→positive, negative→neutral, and neutral→positive vectors. The component of negative→neutral orthogonal to negative→positive exposes additional structure, and Fiction4 projections look banana-shaped/triangular around the three centroids. This suggests a possible **neutral or mixedness residual coordinate** beyond valence polarity.

**Limits.** This is sentence-embedding geometry, not hidden-state causal intervention. Only one embedding model is central; the banana is not fitted as an explicit curve; and neutral structure may reflect topic, style, or corpus composition. The reusable idea is not “a banana has been proved,” but “test whether the neutral residual is a causally independent coordinate under matched-content interventions.”

### 3. Choi and Weber: affective geometry, uncertainty, and a cautionary steering result

**Paper:** Benjamin J. Choi and Melanie Weber, [*Latent Structure of Affective Representations in Large Language Models*](https://arxiv.org/abs/2604.07382) (2026); [author page](https://melanie-weber.com/publication/llm-affect/).

**Code:** the paper's reproducibility statement links an [anonymized Google Drive archive](https://drive.google.com/file/d/1BWcCEOftiBmPMjjL9V61F0yaz5-ZXaff/view?usp=sharing). No stable public GitHub repository was identified.

**Core contribution.** On correctly classified single-label GoEmotions examples, the authors mean-pool hidden states, fit pairwise balanced logistic probes, treat pairwise probe accuracy as a dissimilarity, and reconstruct emotion geometry with multidimensional scaling and Isomap. Gemma-2-9B, Mistral-7B, and Llama-3-70B-Instruct recover a valence–arousal-like organization after Procrustes alignment to human norms.

**Important nuance.** The representations are diffuse and relatively high-rank. Isomap finds modest rank-1 nonlinear improvement, but higher-rank geometry is close to Euclidean and can be approximated linearly. This paper therefore motivates a nonlinear test but also strengthens the linear null.

**Causal appendix.** In Llama-3-70B-Instruct, direct anger↔joy probe steering changes valence but strong negative steering often harms coherence. A “neutral-first” two-segment vector route does not improve human-rated coherence, although a lexical well-formedness diagnostic is slightly better. This is not a learned geodesic: it is a composed vector route with fixed strength. It is best read as evidence that path design and activation support matter, not as a demonstration that manifold steering wins.

**Reusable ideas.** Pairwise discriminability matrices avoid forcing a single global axis; human valence–arousal alignment supplies external coordinates; hyperplane distance and manifold/density distance can be compared as uncertainty predictors; strong negative steering is a stress test for off-manifold collapse.

### 4. Sofroniew et al.: functional emotions and behavior-level causal relevance

**Paper:** Nicholas Sofroniew et al., [*Emotion Concepts and their Function in a Large Language Model*](https://transformer-circuits.pub/2026/emotions/index.html) (Transformer Circuits, 2026); [arXiv archival version](https://arxiv.org/abs/2604.07729).

**Code:** no public experiment repository was identified. The work relies on internal Claude Sonnet 4.5 activation and evaluation infrastructure.

**Core contribution.** The authors derive 171 emotion vectors from synthetic stories in Claude Sonnet 4.5, remove high-variance components estimated from neutral transcripts, and show that the vectors activate in appropriate contexts. Across emotion vectors, PC1 explains 26% of variance and correlates with human valence at about 0.81; another dominant component tracks arousal at about 0.66.

**Causal behavior.** Steering emotion vectors causally changes preferences and alignment-relevant behaviors. Desperation/calm steering strongly changes blackmail and reward-hacking rates, and positive-affect steering shifts a sycophancy–harshness tradeoff. Some dose responses are nonlinear, and valence alone is inadequate: both happy and sad can reduce blackmail in some settings.

**Limits.** The approach explicitly assumes linear vectors and may miss complex mixtures, binding, or key–value-cache mechanisms. It uses one closed model, synthetic/off-policy data, and intervention effects whose downstream mechanism remains incompletely specified.

**Reusable ideas.** A manifold should be judged on behavior, not just sentiment words; valence must be separated from arousal and specific affective states; dose–response curves can reveal a direction leaving its valid support; emotion coordinates can be locally scoped by speaker, token position, and conversational role.

## What the four papers jointly establish—and do not

They jointly establish that:

- a one-dimensional sentiment direction is a strong causal baseline in a controlled generative setting;
- neutral/mixed and affective-category structure may not lie on a single polarity axis;
- nonlinear affective geometry exists descriptively but is often modest and approximately linear;
- affective representations can causally change both local outputs and complex behavior;
- off-manifold steering, confounding, and contextual binding are central failure modes.

They do **not** establish that:

- sentiment activations lie on a particular smooth global manifold;
- a curved path causally outperforms a straight direction;
- a 2D valence–arousal map is intrinsically sufficient;
- neutral-first vector composition approximates a geodesic;
- better reconstruction, probe accuracy, MDS/UMAP/Isomap layout, or human-coordinate alignment implies causal use.

That missing causal comparison is the project's novelty.

## Workspace repository map

All substantive top-level research directories were inspected. They form a useful pipeline from representation discovery to geometry, intervention, and mechanism.

The synthesis refers to the checked-out snapshots below, so later code changes can be distinguished from what was actually inspected:

| Directory | Commit |
|---|---|
| `causalab` | `e433cce` |
| `decomposing-activations-local-geometry` | `ae8aa7c` |
| `sae-manifold` | `f2632dd` |
| `shape-of-beliefs` | `a9d2c03` |
| `block-sparse-featurizer` | `219f121` |
| `arithmetic-wild` | `d03024a` |
| `LLM-addition` | `b586977` |

### `causalab`: the experimental backbone

**Local code:** [`../causalab`](../causalab) · **upstream:** [goodfire-ai/causalab](https://github.com/goodfire-ai/causalab)

The repository already provides the needed analysis dependency chain:

```text
baseline → locate → subspace → activation_manifold → path_steering
    └────────────────────────────→ output_manifold → pullback
```

Most useful modules:

- [`baseline`](../causalab/causalab/analyses/baseline): verifies task accuracy, counterfactuals, and output distributions.
- [`locate`](../causalab/causalab/analyses/locate): layer × token-position interchange scans.
- [`subspace`](../causalab/causalab/analyses/subspace): PCA, DAS, DBM, Boundless DAS, and fixed bases.
- [`activation_manifold`](../causalab/causalab/analyses/activation_manifold): spline/flow fitting over activation centroids.
- [`output_manifold`](../causalab/causalab/analyses/output_manifold): full output distributions embedded with Hellinger geometry.
- [`path_steering`](../causalab/causalab/analyses/path_steering): geometric vs. raw-linear vs. linear-subspace paths, evaluated by coherence, belief-manifold distance, and activation–belief isometry.
- [`pullback`](../causalab/causalab/analyses/pullback): optimizes activation trajectories to realize desired belief-space paths; useful as a rescue method and upper bound.
- [`causal_sufficiency`](../causalab/causalab/analyses/causal_sufficiency), [`ablation`](../causalab/causalab/analyses/ablation), and [`path_patching`](../causalab/causalab/analyses/path_patching): test restoration, necessity, and component-level information flow.
- [`develop_hypothesis`](../causalab/causalab/analyses/develop_hypothesis): checks whether proposed high-level causal models are distinguishable on the planned data.

Two important caveats require extensions. The current activation-manifold analysis uses class centroids, so binary positive/negative data alone cannot identify curvature; the sentiment task must expose several ordered/mixed control points. Also, the shipped path metrics do not replace Tigges's logit recovery/flip outcomes; those outcomes must be added as primary metrics.

### `shape-of-beliefs`: measure behavior on the probability simplex

**Local code:** [`../shape-of-beliefs`](../shape-of-beliefs) · **upstream:** [raphael-goodfire/shape-of-beliefs](https://github.com/raphael-goodfire/shape-of-beliefs)

[`utils/inpca.py`](../shape-of-beliefs/utils/inpca.py) builds pairwise Hellinger distances using square-root probabilities and performs classical MDS. [`linear_field_probes.py`](../shape-of-beliefs/linear_field_probes.py) fits local linear fields over ordered belief states; this is a compelling control because “locally linear, globally curved” may beat both one global direction and a high-capacity opaque manifold. The steering explorer compares global endpoint directions with routes through ordered centroids. Reuse the geometry, but audit intervention scaling before treating the app implementation as a benchmark.

### `sae-manifold`: determine whether dictionaries shatter a manifold

**Local code:** [`../sae-manifold`](../sae-manifold) · **upstream:** [goodfire-ai/sae-manifold](https://github.com/goodfire-ai/sae-manifold) · **paper:** [*Do Sparse Autoencoders Capture Concept Manifolds?*](https://arxiv.org/abs/2604.28119)

[`subspace_capture.py`](../sae-manifold/subspace_capture.py) measures geometric and statistical variance captured by selected SAE decoder atoms against PCA and random baselines. [`unsupervised_clustering.py`](../sae-manifold/unsupervised_clustering.py) clusters features using decoder cosine, coactivation, correlation, mutual information, or inverse-Ising relations. This supports a direct question: is sentiment a single feature/direction, or a cluster whose joint subspace follows a manifold? Cluster bases can be passed into causalab as fixed subspaces.

### `decomposing-activations-local-geometry`: fit a local-linear atlas

**Local code:** [`../decomposing-activations-local-geometry`](../decomposing-activations-local-geometry) · **upstream:** [ordavid-s/decomposing-activations-local-geometry](https://github.com/ordavid-s/decomposing-activations-local-geometry)

The mixture-of-factor-analyzers implementation in [`modeling/mfa.py`](../decomposing-activations-local-geometry/modeling/mfa.py) represents activations as several local affine factor models with posterior responsibilities and held-out likelihood. [`intervention/mfa_steering.py`](../decomposing-activations-local-geometry/intervention/mfa_steering.py) supports pulling activations toward component centroids and moving within local coordinates. MFA is an excellent middle ground: compare `K=1,q=1` (direction-like) with `K>1,q>1` (an atlas), using density/NLL to diagnose unsupported steering.

### `block-sparse-featurizer`: learn concept blocks rather than single features

**Local code:** [`../block-sparse-featurizer`](../block-sparse-featurizer) · **upstream:** [goodfire-ai/block-sparse-featurizer](https://github.com/goodfire-ai/block-sparse-featurizer)

The repository learns groups of directions whose block norm signals concept presence while within-block coordinates encode location. Grassmannian, group-lasso, and vanilla block-sparse variants provide a natural `group_size=1` direction baseline. Its demonstrated domain is vision rather than language, so sentiment use is exploratory and must be validated rather than assumed.

### `arithmetic-wild`: causal coordinates and cross-task transfer

**Local code:** [`../arithmetic-wild`](../arithmetic-wild) · **upstream:** [goodfire-ai/arithmetic-wild](https://github.com/goodfire-ai/arithmetic-wild) · **paper:** [*Arithmetic in the Wild: Llama Uses Base-10 Addition to Reason about Cyclic Concepts*](https://arxiv.org/abs/2605.01148)

[`src/train_das.py`](../arithmetic-wild/src/train_das.py), [`src/cross_task_patch.py`](../arithmetic-wild/src/cross_task_patch.py), and the explicit Fourier-probe code show how to test structured coordinates by both held-out regression and causal cross-task patching. For affect, use psychologically motivated valence/arousal/mixedness coordinates rather than blindly copying periodic Fourier features. The key methodological inheritance is cross-task causal transport.

### `LLM-addition`: a model example of nonlinear causal superiority

**Local code:** [`../LLM-addition`](../LLM-addition) · **upstream:** [subhashk01/LLM-addition](https://github.com/subhashk01/LLM-addition) · **paper:** [*Language Models Use Trigonometry to Do Addition*](https://arxiv.org/abs/2502.00873)

This work fits explicit helices and then compares their causal intervention recovery to PCA, rather than stopping at a geometric fit. It also traces the coordinates through heads, MLPs, neurons, and logits. The transferable standard is: propose an interpretable nonlinear parameterization, validate it on held-out activations, intervene with it, compare against dimension-matched baselines, and explain its circuit.

### Root `docs` and `skills`

The root [`docs`](../docs) contain a careful notebook dependency analysis for the addition/helix work and a subtraction extension plan. They reinforce that representation evidence, answer-position emergence, causal sufficiency, and circuit tracing are distinct stages. The root [`skills`](../skills) document causalab's experiment-planning workflow: objective → falsifiable hypotheses → task causal model → analysis DAG → pre-flight gates → sweep/cache plan. This structure is reflected in the roadmap rather than invoked as an automated skill here.

## Synthesis: the most promising representation family

A sensible search order is:

1. **1D DAS** (exact baseline).
2. **Dimension-matched linear DAS/PCA** (capacity control).
3. **Neutral-residual 2D coordinate system** (Lyngbæk-inspired interpretable candidate).
4. **Valence–arousal or valence–mixedness spline** in a causally located subspace.
5. **Local linear field / MFA atlas** (robust to branching or heterogeneous contexts).
6. **SAE cluster or block-sparse subspace + manifold** (tests feature fragmentation).
7. **Flow or pullback-optimized paths** (only after simpler models fail and with strict held-out density controls).

This ordering makes the simplest falsifiable hypothesis win first and prevents a flexible manifold from absorbing dataset artifacts.
