# Execution roadmap

## Overview

The program is ordered to maximize information per unit of compute and to prevent a flexible manifold search from contaminating the Tigges benchmark.

```text
Phase 0: reproduce Tigges
   ↓ gate: causal baseline matches
Phase 1: build identifiable graded/compositional task
   ↓ gate: counterfactuals and line/curve hypotheses are distinguishable
Phase 2: locate causal cells and measure held-out geometry
   ↓ gate: nonlinear candidate beats dimension-matched linear fit on validation
Phase 3: frozen causal comparison on ToyMovieReview and SST
   ↓ gate: primary superiority/non-inferiority rule
Phase 4: path naturalness, composition, OOD, uncertainty
   ↓ gate: stable advantage and support
Phase 5: circuit and broader behavior
```

## Phase 0: baseline replication

### Work

- Pin an environment for the official Tigges repository.
- Recreate ToyMovieReview, ToyMoodStory, SST pairs, and prompts.
- Reproduce direction methods and layer sweeps on Pythia-1.4B.
- Reproduce SST full-direction and comma-only ablations on Pythia-2.8B.
- Save exact per-pair outputs and activations needed for later baselines.

### Required artifacts

- `replication/environment.md`
- `replication/data_manifest.md`
- `replication/directions/`
- `replication/per_pair_results.*`
- `replication/layer_sweeps/`
- `replication/discrepancies.md`

### Gate

Do not move to a superiority claim until the main method ordering, OOD difficulty, and summarization motif are qualitatively reproduced.

## Phase 1: task and causal abstraction design

### Work

- Add graded valence levels and matched neutral/mixed examples to the toy scaffold.
- Add operator, arousal, topic/template, and optional entity variables factorially.
- Build single-variable counterfactual generators.
- Define exact output tokens and verify tokenizer behavior.
- Partition by adjective/verb/template/source group.
- Use the causal-model distinguishability analysis for line, plane, banana, branch, atlas, and confound hypotheses.

### Minimum viable task

Start with:

- one model: Pythia-1.4B;
- one prompt family matching ToyMovieReview;
- seven valence bins;
- positive/neutral/negative endpoint labels;
- matched negation and contrast subsets;
- adjective, comma/period, and final-token positions.

Do not begin with all 171 emotions or multiple languages.

### Gate

- High baseline accuracy on binary endpoints.
- Adequate coverage per valence/control cell.
- Single-variable counterfactuals pass manual and automated audits.
- Competing causal models predict different interventions on a meaningful fraction of examples.

## Phase 2: representation discovery

### Work package 2A: locate

- Run coarse layer × token-position interchange for `summary_valence`.
- Repeat separately for `operator`, `mixedness`, and `arousal` where used.
- Fine-scan around peaks.

### Work package 2B: linear baselines

- Fit 1D DAS and the remaining Tigges directions.
- Fit DAS/PCA at \(k=2,3,4,8,16\).
- Save validation causality and held-out geometry metrics.

### Work package 2C: simple manifolds

- Valence spline.
- Valence+neutral-residual surface.
- Piecewise linear ordered-centroid route.
- Same-subspace chord controls.

### Work package 2D: local geometry if needed

- MFA `K=1,q=1` through validation-selected `K,q`.
- Local linear field.
- SAE cluster or block-sparse basis only after the activation-subspace study.

### Gate

Advance at most two manifold candidates that:

- improve held-out activation fit/support over a dimension-matched affine model;
- have stable intrinsic coordinates/tangents across bootstraps;
- beat or plausibly complement 1D DAS on validation causality;
- pass shuffled-coordinate and nuisance-confound controls.

If none pass, report the linear result and stop the primary manifold claim.

## Phase 3: preregistered Tigges comparison

### Freeze before running

- model revisions and hook sites;
- layer/token selection rule;
- data groups and locked final-test IDs;
- maximum two manifold candidates;
- baselines and capacity controls;
- residual-handling variant;
- intervention-size matching;
- superiority and non-inferiority margins;
- primary/secondary outcomes;
- bootstrap/permutation procedure and multiplicity families;
- missing/failed-run policy.

### Confirmatory experiments

1. ToyMovieReview held-out adjectives/verbs.
2. Original Tigges SST pair set.
3. SST lexical/template OOD groups.
4. Comma-only and all-token ablation/restoration.

### Primary result table

For every method report:

| Method | Dim. | Toy recovery | Toy flips | SST recovery | SST flips | Specificity | Support violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D DAS | 1 | | | | | | |
| DAS-2D/3D | 2/3 | | | | | | |
| Same-\(k\) linear | \(k\) | | | | | | |
| Manifold chord | \(k\) | | | | | | |
| Manifold geodesic/coordinate | \(k\) | | | | | | |
| Full activation patch | full | | | | | | |
| Random/shuffled | matched | | | | | | |

### Gate and interpretation

- **Full win:** manifold satisfies the frozen superiority rule on SST and is non-inferior or superior on toy.
- **Conditional win:** endpoint non-inferiority plus prespecified path/OOD/composition superiority.
- **Capacity win only:** same-\(k\) linear matches manifold; conclude multidimensional, not nonlinear.
- **No win:** DAS remains the preferred causal abstraction.

## Phase 4: explain utility

Run only after Phase 3 is interpreted.

### 4A. Composition

Test negation, intensifiers, lexical conflict, and contrast/concession with group-held-out operators/templates.

### 4B. Activation–belief geometry

- Fit output manifolds on full vocabulary distributions plus “other” mass.
- Compare geometric, raw-linear, and linear-subspace paths.
- Evaluate Hellinger distance from natural belief support, isometry, coherence, and monotonicity.
- Use pullback as an upper bound, constrained by activation density.

### 4C. Safe dose range

Perform effect-matched strength sweeps in both directions. Blind human raters to direct vs. manifold condition. Focus on negative/high-arousal steering where Choi and Weber found degeneration.

### 4D. Uncertainty

Compare linear margin with manifold reconstruction, geodesic distance, MFA likelihood, and responsibility entropy for model errors and intervention failures.

### 4E. Transfer

Progress in order: new toy templates → SST groups → EmoBank/Facebook → Fiction4/language. Separate zero-shot coordinate transport from target-data alignment/fine-tuning.

## Phase 5: mechanism

### Questions

- Where are local evidence and summary coordinates written?
- Which heads transport them to punctuation/names/final positions?
- Which MLPs transform one local chart to another?
- Which later components read the coordinate into output beliefs?
- Is the circuit shared across positive, neutral/mixed, and negative regions?

### Analyses

- head/MLP activation patching;
- sender→receiver path patching;
- manifold-coordinate ablation and complement sufficiency;
- sparse circuit keep/restore experiments;
- local tangent/factor read-write fits;
- layer-to-layer coordinate transport;
- optional neuron analysis only for a compact validated circuit.

### Gate

A mechanism requires both necessity and sufficiency/restoration. A heatmap or probe-to-component correlation is not enough.

## Phase 6: broader affective behavior

Only after the sentiment result is stable:

- replicate GoEmotions geometry and uncertainty on an open model;
- test actual learned geodesics against Choi and Weber's direct and neutral-first routes;
- test valence/arousal/mixedness coordinate selectivity;
- adapt open behavioral evaluations inspired by Sofroniew et al. for preferences, sycophancy/harshness, or agentic stress;
- test binding by speaker/entity and local operative affect.

Avoid presenting Claude Sonnet 4.5 behavior as directly replicated without access to the model and internal infrastructure.

## Recommended experiment matrix

### Confirmatory core

| Axis | Values |
|---|---|
| Model | Pythia-1.4B; Pythia-2.8B for summarization ablation |
| Dataset | ToyMovieReview; SST |
| Cell | Tigges best DAS cell; independently validation-selected summary cell |
| Representation | DAS-1D; DAS-2D/3D; same-\(k\) linear; max two manifolds |
| Intervention | coordinate replacement; direction patch; full patch; random/normal controls |
| Split | lexical; template; SST source-group |
| Seeds | at least 5 representation/training seeds where stochastic |

### Exploratory extension

| Axis | Values |
|---|---|
| Variables | valence; neutral/mixedness; arousal; operator |
| Geometry | spline; local field; MFA; SAE cluster; block sparse; flow/pullback |
| Path | chord; subspace line; ordered centroids; geodesic; belief pullback |
| Context | simple; negation; intensification; contrast; multi-entity |
| Domain | SST; EmoBank; Facebook; Fiction4; GoEmotions |

## Compute strategy

### Low-cost first

- Verify prompts/pairs/token positions on CPU or tiny model.
- Cache Pythia-1.4B activations once for all offline geometry fits.
- Run causal scans on small stratified debug sets and coarse layers.
- Eliminate models using validation gates before full SST path sweeps.

### Expensive only after gates

- full layer × token scans;
- trained DAS/DBM sweeps;
- all-pair multi-step generation paths;
- normalizing flows/pullback optimization;
- head/MLP/path-patching circuit sweeps;
- 70B or multilingual extensions.

Track GPU-hours per successful and failed configuration. Flexible methods should not receive a much larger tuning budget than linear controls.

## First implementation milestones

1. **Milestone A:** a reproducibility notebook/script that exactly computes Tigges directional recovery and flips for a small fixed pair set.
2. **Milestone B:** a causalab sentiment task reproducing the binary ToyMovieReview baseline.
3. **Milestone C:** graded valence task with explicit causal variables and valid counterfactuals.
4. **Milestone D:** path-steering extension that emits Tigges outcomes, same-subspace controls, and density traces.
5. **Milestone E:** locked pilot comparison of DAS-1D vs. one spline and one MFA/local-field candidate.
6. **Milestone F:** confirmatory Toy→SST result and failure analysis.
7. **Milestone G:** circuit follow-up only if warranted.

## Preregistration-ready headline

> We will conclude that a sentiment manifold is causally more useful than Tigges et al.'s sentiment direction only if, on a locked held-out test set, it improves paired counterfactual behavior or meets prespecified endpoint non-inferiority while improving path support/specificity, survives comparison to a dimension-matched linear subspace and a straight path within the same subspace, and generalizes to a prespecified lexical/template or SST shift. Otherwise, we will retain the one-dimensional direction—or a multidimensional linear subspace—as the preferred causal abstraction.

## Immediate next action

Implement Phase 0 before manifold code. The official baseline repository is public, the primary model is tractable, and the workspace already contains most downstream analysis primitives. The first new code should therefore be a faithful Tigges metric/task adapter for causalab, not a new manifold architecture.
