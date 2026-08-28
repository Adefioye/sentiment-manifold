# Diagnostics and recovery playbook

## The diagnostic principle

A failed causal comparison can mean at least five different things:

1. the representation really is linear;
2. the data do not identify curvature;
3. the manifold model is wrong or overfit;
4. the right representation was measured at the wrong layer/position/site;
5. the representation is real but not causally used, or the intervention leaves its natural support.

Diagnose these possibilities in order. Do not respond to every failure by increasing manifold capacity.

## Gate A: task and counterfactual validity

### Symptoms

- Low or unstable model accuracy before intervention.
- Clean/corrupted examples differ in topic, length, tokenization, syntax, or multiple causal variables.
- Counterfactual target output is not actually preferred by the unmodified model.
- SST pair results depend on a few duplicated phrases or parse-tree relatives.

### Tests

- Manual audit stratified by template/operator/label.
- Exact tokenizer and token-position checks.
- Counterfactual sanity report and pre-intervention output distributions.
- Group leakage audit by adjective, verb, source phrase, template, and document.
- Swap-direction symmetry: source→target and target→source.

### Fixes

- Restrict the causal endpoint set to pairs the model solves in both directions, but report selection and coverage.
- Regenerate minimal pairs changing one variable.
- Match or stratify token length rather than padding away structural differences.
- Split by source group, not by individual generated row.

Do not proceed to geometry until this gate passes.

## Gate B: reproduce the line

### Symptoms

- 1D DAS does not reproduce Tigges's qualitative ordering.
- Mean difference or random directions behave unexpectedly.
- Recovery is strongly asymmetric or sign conventions reverse by layer.

### Tests

- Compare exact official-code activations and prompt strings on a tiny fixed set.
- Unit-test directional replacement algebra.
- Check residual-stream hook location and layer indexing.
- Fix direction sign on validation data only.
- Compare logits before/after full activation patch, then directional patch.

### Fixes

- Resolve model/tokenizer/version differences.
- Align token positions and clean/corrupted pair ordering.
- Separate GPT-2 activation-addition conventions from Pythia directional-patching conventions.

If the baseline cannot be reproduced, manifold superiority is uninterpretable.

## Gate C: is there held-out nonlinear geometry?

### Common false positives

- UMAP/t-SNE visually bends a linear cloud.
- Three centroids form a “banana” because neutral examples have a topic/style shift.
- Curvature appears only after label-conditioned pooling.
- A spline interpolates training centroids exactly but has poor held-out reconstruction.
- A larger subspace improves fit merely because it has more dimensions.

### Required tests

- Fit in the original causally selected subspace; use UMAP only for visualization.
- Compare held-out line, affine plane, polynomial/spline, local field, and MFA likelihood/reconstruction.
- Bootstrap curvature, tangent angles, intrinsic dimension, and chart assignments.
- Residualize or adversarially remove topic/style/length, then repeat.
- Run label/intrinsic-coordinate shuffles through the full fitting pipeline.
- Fit on one lexical/template group and predict held-out groups.
- Compare geometry before and after outlier-norm filtering.

### Interpretation

- **No held-out nonlinear gain:** accept the linear geometric null for this cell/data. Do not run expensive geodesic claims.
- **Nonlinear gain only with nuisance variables:** the geometry is probably a confound, not a sentiment manifold.
- **Stable nonlinear gain but poor causal effect:** proceed to Gate D; descriptive geometry may not be used.

### Repairs when data underidentify geometry

- Add 7–9 ordered valence levels rather than only three broad labels.
- Add matched neutral/mixed examples, not a separate neutral corpus.
- Cross valence with arousal and mixedness to distinguish a curve from a surface.
- Increase lexical/template diversity while keeping causal variables factorially balanced.

## Gate D: geometry exists, but causal interventions do not work

### Possible causes

- Wrong layer, token position, or neural surface.
- Correlational subspace rather than causal subspace.
- The manifold coordinate decodes sentiment but downstream computation does not read it.
- Residual preservation combines an incompatible source residual with a target chart point.
- Endpoint decoder is inaccurate.
- Intervention targets an average state rather than example-specific counterfactual state.

### Tests

- Full activation patch positive control at the same cell.
- 1D DAS at the same cell; if it works, localization is likely correct.
- Causal location scan for each variable separately.
- Fit the manifold inside DAS/DBM/fixed causal subspaces, not just PCA.
- Compare centroid target, paired target-coordinate, and decoded-coordinate interventions.
- Test source-residual preserve/transport/replace variants.
- Measure downstream coordinate propagation at later layers.
- Restore manifold coordinates after upstream corruption (`causal_sufficiency`).

### Fixes

- Initialize the representation subspace with causal DAS, then fit only its internal geometry.
- Add a validation causal-alignment term while retaining a locked test set.
- Use a local atlas/MFA instead of a global decoder when residual compatibility depends on context.
- Intervene at the summary position rather than all tokens.
- Use pullback to determine whether any supported activation path can realize the target belief. If pullback fails, the chosen cell may not control the behavior.

## Gate E: causal effect exists, but the manifold does not beat DAS

### Possible explanations

- The endpoint task is genuinely one-dimensional.
- Binary endpoints saturate, leaving no room for improvement.
- Extra manifold dimensions add nuisance and hurt OOD generalization, as Tigges's DAS-2D/3D result suggests.
- The geodesic and chord are almost identical because curvature is modest.
- Gains occur only on intermediate states, not endpoint flips.

### Tests

- Same-subspace geodesic vs. chord.
- Restrict to preregistered neutral/mixed/compositional stress subsets.
- Endpoint non-inferiority plus path/support superiority.
- Plot gain against estimated local curvature and density; a real mechanism should show larger gains where the line is least adequate.
- Compare `K=1,q=1` MFA against richer atlas models.

### Valid conclusions

- **DAS wins everywhere:** sentiment is causally line-like in tested settings.
- **Tie at endpoints, manifold wins supported intermediate trajectories/OOD:** manifold is more useful for graded control, not for binary classification.
- **Manifold wins only high-curvature subsets:** report a conditional representation, not a universal replacement.
- **Higher-dimensional linear control matches manifold:** the gain is multidimensionality, not curvature.

Do not redefine “beat” after seeing which of these occurred.

## Gate F: steering becomes incoherent or degenerate

### Diagnostics

- Activation L2 and Mahalanobis distance from natural states.
- kNN support, MFA NLL/responsibility entropy, decoder normal residual.
- Full-vocabulary Hellinger distance and probability assigned to “other” tokens.
- Repetition, profanity, broken syntax, refusal, semantic drift.
- Target-effect-matched direct vs. manifold comparison.
- Per-step Jacobian condition number and tangent norm.

### Likely failure patterns and fixes

| Failure | Diagnosis | Recovery |
|---|---|---|
| Strong direction exits support | Density falls before coherence | Smaller supported steps; projection/replacement; local atlas; stop at support boundary |
| Manifold decoder folds/self-intersects | Nearby intrinsic points decode far apart; unstable inverse | Increase ambient \(k\); regularize Jacobian; use MFA/atlas or injective flow |
| Over-smoothed spline cuts corners | Geodesic traverses low-density regions | Tune smoothness on validation density; use data-graph geodesic/local field |
| Exact spline chases centroids | Good train, poor held-out | Regularize; more control points/data; nested validation |
| Negative steering collapses earlier | Density/RLHF asymmetry | Direction-specific step schedules; local charts; effect matching; report asymmetric reachable set |
| “Neutral-first” does not help | Waypoint vectors still leave the natural surface | Learn an actual path from natural activations/output beliefs; match total arc length and effect |
| Pullback finds odd adversarial states | Low belief loss, poor activation support | Add activation-density/energy constraint; constrain to atlas/geodesic tube |

## Specific representation failure modes

### Neutral coordinate is actually topic/style

**Signal:** neutral-residual score predicts corpus/domain better than neutral behavior; causal effect disappears in matched templates.

**Fix:** construct same-content triples; residualize nuisance variables using training data; adversarially enforce domain invariance; use within-template centering; validate cross-domain coordinate transport.

### Valence–arousal axes rotate across layers

**Signal:** Procrustes alignment is good but coordinate signs/angles are unstable; coordinate-specific interventions cross-talk.

**Fix:** estimate layer-specific charts linked by orthogonal/CCA transport; do not assume one ambient direction across layers; test whether the changing chart is a functional transformation.

### Manifold is high-rank/diffuse

**Signal:** participation ratio remains high; low-rank reconstruction plateaus; Choi-style pairwise geometry is stable but no 2D decoder suffices.

**Fix:** distinguish low-dimensional *semantic organization* from low-dimensional *activation support*. Fit local task-relevant coordinates inside a higher-rank nuisance space; preserve nuisance residuals; use local factor analyzers.

### SAE representation is shattered

**Signal:** no single feature captures sentiment, but several decoder atoms coactivate and collectively explain variance.

**Fix:** cluster by decoder geometry/coactivation/MI; use cluster decoder span as a fixed causalab subspace; compare to random same-size clusters and PCA; consider a block-sparse featurizer.

### Atlas components track templates rather than affect regions

**Signal:** MFA responsibility is predicted by surface template/domain, not intrinsic affect state; component steering changes style.

**Fix:** balance templates within every affect cell; add nuisance-invariant regularization; test held-out templates; condition local chart on causal context only if that context is part of the declared abstraction.

### Manifold varies by entity/speaker

**Signal:** steering one character changes another or loses effect after intervening at a distant token.

**Fix:** model sentiment and binding separately; locate speaker/entity-specific carrier positions; test tensor/product or key–value binding rather than forcing everything into one affect surface.

## Negative and positive controls

Every confirmatory experiment should include:

- random direction/subspace/manifold with matched dimension and smoothness;
- label and intrinsic-coordinate shuffle;
- same-subspace straight chord;
- dimension-matched linear DAS/PCA;
- equal-norm tangent and normal displacement;
- full target activation patch;
- no-op/source-to-source patch;
- unrelated causal variable intervention;
- reversed path and swap-direction symmetry;
- natural target activations as a density/support reference.

The shuffle must rerun model selection. Shuffling only after fitting underestimates flexible-pipeline false positives.

## Sensitivity grid

At minimum vary:

- seed and train-size;
- layer/token position/hook site;
- \(k\)-dimensional ambient subspace;
- number/distribution of intrinsic control points;
- spline smoothness, MFA \(K,q\), or local-neighbor scale;
- residual handling;
- path step count and intervention strength;
- pooling vs. exact token position;
- label-token choice and full-vocabulary metric;
- human-rating source and neutral thresholds.

Report a specification curve or multiverse summary. The conclusion should not depend on one narrow, post-hoc configuration.

## Stopping rules

- Stop geometry expansion if no candidate beats a dimension-matched linear model on held-out reconstruction *and* validation causality.
- Stop circuit work if the winning representation has no stable causal advantage across seeds/splits.
- Stop increasing model capacity when shuffled-coordinate controls also improve.
- Stop extrapolating a path when its support score leaves the prespecified natural percentile.
- Declare the linear null supported if two well-powered confirmatory attempts across the primary model/datasets fail the frozen superiority rule.

These rules turn failure into a result rather than an invitation to tune indefinitely.
