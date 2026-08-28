# Experimental methodology

## 1. Separate replication, discovery, and confirmation

Use three phases with hard data boundaries.

1. **Replication:** reproduce Tigges's public code, datasets, prompts, direction methods, and main numbers. Do not introduce a manifold yet.
2. **Discovery:** use training/validation data to construct graded sentiment states, choose causal variables, locate cells, select intrinsic dimension/model family, and diagnose failures.
3. **Confirmation:** run the preregistered winning manifold and frozen baselines once on locked ToyMovieReview and SST test sets, then on prespecified OOD/compositional splits.

Never select layers, control points, spline smoothness, MFA components, step size, or superiority metric from final-test results.

## 2. Stage 0: exact Tigges replication

### Models

- GPT-2-small for ToyMovieReview activation addition and direction visualization.
- Pythia-1.4B for the primary ToyMovieReview→SST directional-patching comparison.
- Pythia-2.8B for SST zero-shot classification, directional ablation, and comma summarization.
- Optionally reproduce smaller/larger Pythia scaling only after the principal models match.

Record exact model revisions, tokenizer revisions, dtype, TransformerLens version, device, prompt strings, and random seeds.

### Data and splits

- Recreate ToyMovieReview from the official repository, including the 55/30 adjective train/test split and controlled verbs.
- Recreate the exact ToyMoodStory prompts as a secondary validation.
- Reproduce Tigges's SST filtering, binary collapse, token-length pairing, prompts, and subset on which the model is initially correct.
- Preserve a raw copy of split identifiers and hashes. Any later graded annotations must be joined without changing the original binary test membership.

### Direction methods

At every candidate layer and specified token position, fit:

- normalized positive-minus-negative mean difference;
- K-means centroid difference;
- L2 logistic-regression weight;
- PCA first component, with sign fixed on validation data;
- 1D DAS;
- 2D and 3D DAS exactly as in the paper;
- random directions matched in count and norm.

### Intervention semantics

For source activation \(h_s\), target activation \(h_t\), and unit direction \(d\), Tigges directional patching is:

\[
h'_s = h_s + d\,d^\top(h_t-h_s).
\]

This preserves the component orthogonal to \(d\). Compute:

\[
\text{recovery}=100\times\frac{\Delta(h'_s)-\Delta(h_s)}{\Delta(h_t)-\Delta(h_s)},
\]

where \(\Delta\) is the paper's positive-minus-negative target logit difference, with sign/orientation standardized. Also report whether the intervention flips the clean decision to the target decision.

Do not clip recovery at 100%; overshoot is informative.

### Replication gate

Proceed only if:

- task baselines and pair counts match the paper/code or discrepancies are explained;
- DAS is the strongest or comparably strong 1D causal method;
- the ToyMovieReview→SST generalization ordering is qualitatively reproduced;
- full-direction SST ablation and comma-only ablation recover the reported qualitative gap;
- random controls are near zero.

Archive deviations rather than silently adapting the benchmark.

## 3. Make curvature identifiable

Binary positive/negative endpoints do not identify a curve: infinitely many paths connect two points. Keep the original binary endpoint evaluation but enrich *training and diagnostic data* with ordered and crossed states.

### Graded ToyMovieReview

Assign continuous valence/arousal/dominance ratings to adjectives and verbs using external human norms where available, supplemented by blinded human ratings. Create 7–9 prespecified valence bins, including a narrow neutral bin, while preserving the original positive/negative labels.

Generate prompts by factorially crossing:

- adjective valence/intensity;
- verb valence/intensity;
- agreement vs. conflict between adjective and verb;
- negation (`not`, `hardly`, `not at all`);
- intensifier/downtoner (`very`, `slightly`, `somewhat`);
- contrast/concession (`but`, `although`, `despite`);
- content/topic and template;
- optional arousal at matched valence.

Hold prompt length/tokenization fixed where possible. When exact length matching is impossible, stratify by token count and include it as a nuisance variable rather than pooling blindly.

### SST and natural text

Use SST phrase-level labels as continuous/ordinal training information only within training/validation partitions. Define stress subsets before final evaluation:

- negation;
- contrastive conjunctions;
- multiple sentiment-bearing spans;
- neutral/midpoint phrases;
- high lexical conflict;
- held-out sentiment lexemes;
- held-out parse/template families.

Later extensions can use EmoBank continuous VAD, Facebook valence/arousal, and Fiction4 multilingual/literary data. GoEmotions belongs to the emotion-category extension, not the primary binary benchmark.

### Counterfactual construction

Each pair should change one high-level variable while holding the others fixed. Prefer generated or curated minimal sets over arbitrary nearest neighbors.

Example causal variables:

- `lexical_evidence`: signed local adjective/verb evidence;
- `operator`: identity/negation/intensifier/concession;
- `summary_valence`: continuous or ordinal final sentiment;
- `mixedness`: amount of conflicting positive/negative evidence;
- `arousal`: low/high or continuous intensity;
- `topic`, `style`, `template`, `length`: nuisance variables;
- `entity`/`speaker`: binding variable for multi-subject tasks;
- `output_label` or output-distribution state.

An initial causal graph is:

```text
topic/style/template ───────► surface form ─────► activations
lexical evidence ───────────► surface form
operator ───────────────────► composed/summary valence ─► output belief
lexical evidence ───────────► composed/summary valence
mixedness/arousal ──────────► composed affect state ─────► output belief
entity/speaker ─────────────► binding of the summary state
```

Use causalab's hypothesis-development machinery to verify that the line, plane, banana, branch, atlas, and confound causal models imply distinguishable counterfactuals on the generated data.

## 4. Activation collection and localization

### Neural surfaces

Collect at least:

- residual stream before/after each block;
- attention output and MLP output for later circuit work;
- Tigges's adjective/verb positions;
- operator token;
- comma/name/period and other summary positions;
- final prompt token immediately before the label/completion.

Avoid mean pooling as the primary representation because it erases the summarization motif. Use pooling only as a secondary comparison to Choi and Weber.

### Causal location scan

In causalab:

1. `baseline` checks model accuracy, counterfactual validity, and class output distributions.
2. `locate` runs pairwise interchange for a named resampled variable, and centroid mode as a broad scan.
3. Scan layers coarsely, then every layer near causal peaks.
4. Select cells on validation counterfactual effect, not probe accuracy.

Maintain separate best cells for lexical valence, operator, summarized valence, mixedness, and arousal. A single alphabetically auto-discovered `best_cell` is unsafe for multi-variable sweeps; use explicit artifact paths.

## 5. Representation candidates

Fit all candidates on the same training activations and select with nested validation.

### Linear baselines

1. 1D DAS (primary baseline).
2. Mean difference, K-means, LR, PCA-1.
3. DAS/PCA with \(k\in\{2,3,4,8,16\}\).
4. A piecewise linear path through ordered centroids.

The dimension-matched linear subspace is essential: it distinguishes gains from extra capacity from gains due to curvature.

### Interpretable manifolds

1. **One-dimensional spline:** intrinsic coordinate is continuous valence; fit a cubic/TPS decoder from valence to activation subspace.
2. **Neutral-residual coordinates:** one coordinate is positive–negative valence; a second is the negative→neutral residual orthogonal to the valence axis.
3. **Valence–mixedness surface:** signed output plus conflicting-evidence strength.
4. **Valence–arousal surface:** human VAD coordinates with an affine, spline, or conformal decoder.
5. **Branched/atlas model:** separate positive, negative, and mixed/neutral local charts joined probabilistically.

### Data-driven manifolds

1. causalab spline/normalizing-flow manifold in a causally selected subspace;
2. local linear field over ordered class/quantile centroids;
3. mixture of factor analyzers with component responsibilities and factor dimension \(q\);
4. clustered SAE or block-sparse basis followed by a low-dimensional manifold;
5. pullback-optimized activation path as a behavior-matching upper bound.

Use a flow or pullback only when simpler candidates fail; their flexibility raises overfitting and interpretability costs.

### Model-selection criteria

Use held-out validation data to measure:

- activation reconstruction and normal residual;
- likelihood/kNN support for probabilistic models;
- intrinsic-coordinate prediction and monotonicity;
- local tangent stability under bootstrap;
- validation causal recovery and specificity;
- model complexity (effective parameters/components) and seed stability.

Geometric fit alone must not choose the final method.

## 6. Fair manifold interventions

Let an encoder infer \(z=E(h)\), a decoder map \(D(z)\) back to the learned activation subspace, and let \(P\) be the ambient subspace projector.

### Coordinate replacement

For target intrinsic coordinate \(z_t\):

\[
h' = (I-P)h_s + D(z_t).
\]

This is the closest analogue of Tigges's directional replacement: it changes the modeled feature and preserves the source residual outside the representation subspace.

### Residual-handling ablations

Run three variants because the orthogonal residual may contain sentiment or may be incompatible with the target chart:

1. preserve source residual;
2. transport/rotate residual using local tangent alignment, if well-defined;
3. replace with a matched target or local-centroid residual.

The primary variant should be fixed before test results. Report whether conclusions depend on residual handling.

### Path steering

For endpoints \(z_s,z_t\), compare:

- Tigges 1D interpolation;
- raw ambient chord;
- straight line in the same \(k\)-dimensional subspace;
- piecewise linear ordered-centroid route;
- manifold geodesic;
- belief-space pullback path.

Evaluate identical endpoint pairs and numbers of intermediate steps. In causalab, use random/unbiased pairs for aggregate inference; selected pairs are visualization examples only.

### Dose and norm matching

Report three matched analyses:

- equal activation L2 and Mahalanobis displacement;
- equal intrinsic/geodesic arc length;
- equal achieved target-logit change.

Also report natural activation percentile, MFA NLL or kNN distance, and decoder reconstruction at every step. This distinguishes “safer path” from “weaker intervention.”

## 7. Outcome metrics

### Primary Tigges metrics

- paired logit-difference recovery;
- paired target logit flip;
- endpoint task accuracy after ablation/restoration;
- comma-only and all-token necessity effects.

### Causal selectivity

- target-coordinate change divided by off-target-coordinate change;
- valence change at fixed arousal/mixedness, and vice versa;
- effect on matched non-sentiment label pairs;
- content-token log-probability drift;
- perplexity/next-token KL on neutral continuations;
- syntax and semantic-content preservation.

### Path quality

- monotonicity of expected sentiment/valence along steps;
- path success area under the target-probability curve;
- coherence probability mass over task-valid output concepts;
- Hellinger distance to the natural belief manifold;
- activation–belief geodesic isometry;
- maximum and integrated support penalty;
- endpoint recapitulation and overshoot.

### Geometry and support

- cross-validated reconstruction and likelihood;
- intrinsic-dimension estimates with uncertainty;
- local tangent-angle consistency;
- geodesic/chord ratio and curvature with bootstrap CIs;
- trustworthiness/continuity for visualization methods;
- nearest-natural-activation distance;
- normal residual and density responsibility entropy.

### Natural-language steering

- blinded human valence and coherence ratings;
- task-specific automated sentiment score, calibrated against human labels;
- lexical diversity, repetition, profanity, refusal, and content retention;
- effect-matched comparisons, especially for negative steering.

LLM judges may be used only with validated rubrics, randomized/blinded conditions, and a human-rated subset.

## 8. Statistical analysis

### Unit of analysis

The counterfactual source–target pair is the unit, not each path step. Path steps are repeated measures within a pair. Cluster by template/source sentence and, for natural text, by original sentence/document.

### Splits

Use group splits to prevent leakage:

- adjective/verb identity;
- source sentence/phrase tree;
- template/operator family;
- topic/domain;
- optionally time/language.

Use nested train/validation/test. The final test is opened once.

### Tests

- Paired bootstrap or permutation CIs for recovery, path scores, and specificity.
- McNemar or paired bootstrap for logit flips/accuracy.
- Hierarchical/mixed-effects models across pairs, seeds, layers, and models.
- Benjamini–Hochberg control for prespecified secondary families.
- Equivalence/non-inferiority tests when endpoint saturation makes raw superiority unrealistic.
- Report standardized and raw effect sizes, CIs, and all seeds—not only p-values.

### Superiority rule

Use a lexicographic or Pareto rule rather than a post-hoc weighted average:

1. endpoint recovery superiority, or non-inferiority within a frozen margin;
2. no meaningful loss in flip rate;
3. specificity/support superiority;
4. OOD/compositional superiority.

If methods trade off, show the Pareto frontier. Do not declare a win from a composite whose weights were chosen after results.

### Multiple models and layers

Treat model and layer as replication axes. A method selected at a different best layer has extra selection flexibility; therefore report both:

- same-cell comparisons at Tigges's best DAS cell;
- independently validation-selected best-cell comparisons with nested selection.

## 9. causalab implementation map

### New task package

Create a future `causalab/causalab/tasks/sentiment_manifold/` package with:

- `causal_models.py`: line, valence–mixedness, valence–arousal, and operator-composition DAGs;
- `templates.py`: exact Tigges templates plus graded/compositional extensions;
- `counterfactuals.py`: single-variable matched pairs;
- `token_positions.py`: adjective, verb, operator, name, comma, period, final label position;
- `config.py`: value grids, train/test sizes, output tokens;
- checker/metrics for exact output labels and continuous belief expectations.

The initial output mode should be single-token labels (`Positive`, `Neutral`, `Negative`) only if tokenizer checks pass. Keep a binary `good`/`bad` or `Positive`/`Negative` mode for exact Tigges comparability.

### Analysis DAG

```text
baseline
  ├─► output_manifold ───────────────────────────────────┐
  └─► locate ─► subspace ─► activation_manifold ─► path_steering
                      │               │                  │
                      ├─► ablation     └─────────────────► pullback
                      ├─► causal_sufficiency
                      └─► path_patching / circuit analysis
```

### Required extensions

- Add Tigges logit-recovery/logit-flip criteria to `path_steering`.
- Add continuous/ordinal intrinsic controls rather than only per-class centroid coordinates.
- Add explicit dimension-matched `linear_subspace` and same-subspace chord reports to every result table.
- Add activation-density/support metrics per path step.
- Add tangent vs. normal intervention controls.
- Add representation-coordinate ablation/keep-complement, not just whole head/MLP ablation.
- Add group-aware dataset splits and pair-level bootstrap output.
- Make all multi-variable artifact paths explicit to avoid auto-discovery selecting the wrong variable.

### Suggested sweep axes

Run separate, collision-safe experiment roots when target variable or dataset split changes. Within a root, safe conceptual sweeps are:

- subspace method: DAS/PCA/fixed SAE cluster;
- \(k\): 1, 2, 3, 4, 8, 16;
- manifold type: line/spline/MFA/local field/flow;
- smoothness or number of MFA components;
- layer/token cell;
- residual handling;
- intervention strength.

Use small debug data and three coarse layers first. Cache baseline activations/output distributions and reuse them across geometry fits.

## 10. Circuit methodology after representation success

Only pursue full circuit work after a manifold method wins or reveals a stable qualitative distinction.

1. **Emergence:** measure layerwise coordinate fit and causal effect at each token position.
2. **Component patching:** patch attention vs. MLP outputs and individual heads.
3. **Path patching:** identify senders carrying lexical/operator information into summary positions and receivers carrying manifold coordinates toward logits.
4. **Ablation/sufficiency:** keep only the candidate circuit or restore its manifold coordinate after upstream corruption.
5. **Local read/write analysis:** regress component outputs against intrinsic tangent fields or MFA factor coordinates; intervene with the fitted component contribution.
6. **Output closure:** use logit lens/full output distributions to show how later components read coordinates into sentiment-relevant tokens.

The mechanism claim requires causal restoration/ablation, not correlations between component activations and coordinates.

## 11. Reproducibility artifacts

Every run should save:

- immutable data split IDs and generation templates;
- model/tokenizer revisions and environment lock;
- activation cache metadata including hook site and token position;
- learned directions/subspaces/manifold checkpoints;
- intrinsic-coordinate definitions and sign/orientation conventions;
- validation selection tables for layer, dimension, smoothness, and strength;
- per-pair predictions/intervention metrics, not just aggregates;
- density/support traces along paths;
- bootstrap indices or seeds and statistical scripts;
- representative successes and failures chosen by a frozen rule;
- resolved causalab configs and artifact dependency manifest.

This makes negative and failed-manifold results reusable rather than disposable.
