# Research questions and falsifiable hypotheses

## Primary question

### RQ1. Does a sentiment manifold causally outperform Tigges's 1D DAS direction in Tigges's own setting?

**Question.** On held-out ToyMovieReview and SST clean/corrupted pairs, can an intrinsic-coordinate intervention on a learned sentiment manifold recover more of the counterfactual logit difference and flip more labels than the best-layer 1D DAS direction, without increasing off-target effects or leaving natural activation support?

**Why this is primary.** It makes the new claim commensurable with the strongest existing causal result. A new dataset or model cannot substitute for winning this test.

**H1.** The manifold will be non-inferior on the toy endpoint task and superior on SST, especially for neutral, mixed, or compositionally complex phrases whose states are poorly captured by polarity alone.

**Falsified if:** after nested model selection and matched intervention budgets, 1D DAS has equal or better held-out recovery, flip rate, specificity, and support; or any manifold advantage disappears against a dimension-matched linear subspace or the straight chord in the same subspace.

**Necessary contrasts.**

- 1D DAS, mean difference, K-means, LR, PCA, random direction.
- DAS-2D and DAS-3D, because Tigges already reports that extra dimensions can overfit.
- Same-\(k\) straight affine path vs. learned geodesic.
- Full activation patch as an upper bound and equal-norm normal displacement as a negative control.
- Label/intrinsic-coordinate shuffle as a flexible-model null.

**Primary outcomes.** Paired logit-difference recovery, paired logit flip, and a specificity/support panel. Path metrics are secondary unless binary endpoints saturate.

## Questions that explain when and why a manifold wins

### RQ2. Is the apparent curvature a genuine valence path or a neutral/mixedness branch?

**Question.** Does the component of negative→neutral that is orthogonal to negative→positive encode causally independent neutral/mixed sentiment, or is it a domain/topic/style confound?

**Motivation.** This is the causal version of Lyngbæk et al.'s banana-shaped observation.

**H2.** At fixed content, intervening on the neutral-residual coordinate will selectively change `Neutral` probability, uncertainty, or mixed-evidence resolution while leaving signed valence approximately fixed. A matched off-manifold displacement will not.

**Falsified if:** the coordinate ceases to exist after matching topic/style/length; only predicts corpus identity; or its intervention simply moves along ordinary valence.

**Decisive design.** Build triples with identical propositional content and controlled positive, neutral/mixed, and negative realizations. Cross negative→neutral and neutral→positive with a separate valence-strength ladder. Compare a 1D line, 2D valence+neutral residual, and a single curved 1D path.

### RQ3. Does a manifold improve causal sentiment composition under negation, intensification, and contrast?

**Question.** Can manifold coordinates preserve sentiment operations such as `not good`, `not bad`, `very good`, `slightly bad`, and `good acting but terrible story` more faithfully than a global direction?

**H3.** A global direction will work for simple lexical substitution, while an atlas or 2D manifold will gain most on negation scope, mixed evidence, and contrast/concession, where the same lexical polarity maps to different summarized sentiment.

**Falsified if:** the direction generalizes equally well across held-out operators/templates, or manifold gains are explainable by extra dimension/capacity rather than curvature.

**Causal test.** Intervene on summarized sentiment while holding lexical evidence and operator variables fixed; separately interchange the operator while holding base valence fixed. The high-level causal model should predict both target output and intermediate activation coordinate.

### RQ4. Where in the network is any manifold advantage created and consumed?

**Question.** Is nonlinear structure strongest at valenced words, at summary positions such as commas/names/periods, or near output readout—and does causal usefulness peak at the same locations?

**H4.** Early positions primarily encode local lexical valence; intermediate layers at punctuation/name summary tokens form a lower-dimensional composed sentiment state; later positions map that state to output beliefs. Manifold gains, if real, should concentrate at summary positions rather than everywhere.

**Falsified if:** curvature has no stable layer/position localization, is present only in pooled activations, or cannot be linked to counterfactual effect.

**Tigges link.** Replicate comma-only directional ablation, then compare tangent/manifold-coordinate ablation at commas with all-token ablation and with matched linear subspace ablation.

### RQ5. Is valence–arousal causally factorized, or does “sentiment” collapse multiple affective variables?

**Question.** Can valence be changed while arousal is held fixed, and arousal changed while valence is held fixed, using intrinsic coordinates more selectively than a polarity direction?

**H5.** A 2D valence–arousal surface will reduce cross-talk: valence intervention changes positive/negative output without changing intensity lexicon; arousal intervention changes intensity/urgency without flipping valence. Benefits should be larger on high-arousal negative states, where straight steering often degenerates.

**Falsified if:** coordinate interventions have the same cross-effects as linear controls; external VAD alignment exists but interventions do not respect it; or the data support a diffuse high-rank representation rather than a stable 2D surface.

**Scope rule.** This question follows, rather than replaces, the polarity benchmark. It should use ANEW/EmoBank-style human ratings and matched generated counterfactuals.

### RQ6. Is sentiment globally curved, locally linear, or a mixture of context-specific affine regions?

**Question.** Which representation class best predicts held-out activations and causal outcomes: a global line, global linear subspace, smooth spline/flow, local linear field, or mixture of factor analyzers?

**H6.** The strongest model will be locally linear but globally curved: multiple low-rank regions with smooth or probabilistic transitions, rather than one global spline through all data.

**Falsified if:** `K=1,q=1` MFA or 1D DAS matches richer models; the selected number of components is unstable; or higher held-out likelihood does not translate into causal gains.

**Model-selection principle.** Select geometry only using training/validation activation likelihood, reconstruction, and preregistered validation causality. Test causal superiority once on a locked set.

### RQ7. Does an activation geodesic better track the model's belief-space path than a straight direction?

**Question.** As sentiment is steered from negative to positive, does the activation-manifold geodesic produce a smoother, more coherent, and more nearly geodesic trajectory in the model's *full output distribution*?

**H7.** At equal arc length/intervention energy, the manifold path will have lower distance from the natural output-belief manifold, higher activation–belief geodesic distance correlation, more monotone target probability, and less unrelated vocabulary drift.

**Falsified if:** output-distribution paths are indistinguishable; improvements exist only for a cherry-picked token pair; or the same gains occur for a straight path in the learned subspace.

**Why full distributions matter.** A positive-minus-negative logit difference can improve while probability mass leaks to incoherent or unrelated tokens. Hellinger geometry with an explicit “other vocabulary” mass detects this.

### RQ8. Does manifold-aware steering extend the safe causal dose range?

**Question.** Does following supported tangent/geodesic paths delay the incoherence, repetition, profanity, or semantic collapse caused by strong activation addition along a global direction?

**H8.** Manifold projection/replacement will preserve coherence at larger effective sentiment changes, particularly for negative/high-arousal steering, because it avoids unsupported normal displacement.

**Falsified if:** degeneration occurs at the same target sentiment or activation Mahalanobis distance; the manifold merely reduces effective intervention strength; or direct vector steering is more coherent at matched behavioral effect.

**Dose matching.** Compare methods at matched target logit change, matched activation Mahalanobis distance, and matched arc length. Fixed raw coefficients alone are not fair.

### RQ9. Does geometry improve transport across lexicon, templates, domains, languages, and tasks?

**Question.** Does a manifold learned on Tigges's toy adjectives/verbs transfer more causally than 1D DAS to held-out adjectives, templates, SST, continuous valence corpora, and eventually another language?

**H9.** The direction will be competitive for ordinary polarity, while a low-dimensional atlas will gain on domain shifts that contain neutral/mixed or compositional structure. Intrinsic coordinates should align across domains more reliably than ambient tangent vectors.

**Falsified if:** transport gains vanish after matched content and label distribution; manifold alignment requires target-test labels; or the CVP-style direction transfers better.

**Datasets in order.** ToyMovieReview held-out lexicon → new held-out templates → SST → EmoBank/Facebook → Fiction4/Danish. The latter three are external extensions, not replacements for SST.

### RQ10. Do distance-to-manifold and local density diagnose sentiment uncertainty better than linear margin?

**Question.** Do geodesic distance, normal reconstruction error, MFA likelihood, or local responsibility entropy predict model mistakes and steering failures beyond distance to the Tigges/probe hyperplane?

**H10.** Hyperplane margin will capture binary confidence; manifold distance/density will add predictive value for neutral, mixed, mislabeled, OOD, and soon-to-collapse interventions.

**Falsified if:** nested held-out models show no incremental AUC/calibration gain; density scores are dominated by length/topic; or the same signal is obtained from activation norm.

**Causal extension.** Move equal distances tangent to and normal to the manifold. If normal distance is genuinely diagnostic, normal interventions should degrade coherence more than tangent moves at comparable sentiment effect.

### RQ11. Is the sentiment manifold represented by clusters or blocks of features rather than a single feature?

**Question.** Do SAE feature clusters or block-sparse groups capture a causally useful sentiment manifold that a single SAE feature or direction misses?

**H11.** Sentiment-related SAE atoms will form coactive/geometrically related clusters; a cluster-derived fixed subspace followed by a manifold model will outperform the best single atom and match or exceed DAS with better interpretability.

**Falsified if:** selected clusters are unstable, do not beat random/orthogonal groups, or improve variance explained without improving causal intervention.

**Controls.** PCA optimal capture, random orthogonal directions, random overcomplete dictionaries, same group size/sparsity, outlier-norm filtering, and feature-cluster bootstrap stability.

### RQ12. Which heads and MLPs implement manifold coordinates and their transformations?

**Question.** Can a small circuit explain how local sentiment evidence becomes a summarized manifold coordinate and then a belief/output change?

**H12.** Specific heads transport local valence/operator information to summary positions; middle MLPs construct the manifold coordinate; later components read tangent/local coordinates into logits. Different atlas regions may use partly distinct components.

**Falsified if:** component patching effects do not track coordinate changes; retaining the proposed circuit fails to preserve behavior; or ablating it has no selective effect.

**Method.** Follow the `LLM-addition` evidence chain: representation fit → answer/summary-position emergence → whole-layer/head/MLP activation and path patching → sparse circuit sufficiency → optional neuron-level read/write analysis.

### RQ13. Are sentiment-coordinate interventions path independent?

**Question.** Do sequential interventions commute—for example, valence then arousal vs. arousal then valence—and do closed loops return both activations and beliefs to the start?

**H13.** A sufficient static 2D manifold should be approximately path independent at a fixed layer. Strong order effects or closed-loop drift imply missing state, context binding, layer dynamics, or an atlas with transitions rather than a single coordinate chart.

**Falsified if:** after controlling intervention energy and layer, both orders and closed loops agree within a prespecified tolerance.

**Value even if falsified.** Approximate commutativity would support the causal abstraction. Noncommutativity would reveal a deeper mechanism than a static sentiment surface.

### RQ14. Are affective coordinates bound to the correct speaker, entity, and time?

**Question.** Can manifold interventions alter one character's or speaker's sentiment without changing another's, and can summarized coordinates persist or be retrieved at the correct later token?

**H14.** A content-only sentiment manifold will fail some multi-entity tasks unless combined with binding variables or token-position-specific retrieval. This may explain why apparently universal directions have incomplete causal recovery.

**Falsified if:** the same manifold coordinate transfers selectively across roles/entities with no binding augmentation.

**Link to the papers.** Tigges's ToyMoodStory and name-token summarization provide the baseline; Sofroniew et al.'s locally operative speaker-dependent emotion vectors motivate richer dialogues.

## Competing causal abstractions

These should be specified before experiments and checked for observational/interventional distinguishability:

1. **Line model:** one scalar signed valence causes the summary output.
2. **Linear-plane model:** valence plus arousal or neutral/mixedness, with an affine decoder.
3. **Banana model:** one curved ordered coordinate passes through negative, neutral, and positive states.
4. **Branched model:** positive and negative arms share a neutral/mixed junction; one scalar is topologically insufficient.
5. **Local-atlas model:** context/operator/domain selects a local affine chart, then low-dimensional affect coordinates act within it.
6. **Confound model:** apparent curvature is caused by topic/style/length/lexicon, while the causal sentiment variable remains linear.
7. **Distributed circuit model:** no compact manifold is causally sufficient; affect is dynamically reconstructed from multiple features and bindings.

The experiment should be able to return “line” or “confound” as the winner. Otherwise it is not a fair test of manifolds.

## Priority tiers

### Tier A: paper-defining

- RQ1 exact causal superiority.
- RQ2 neutral/mixedness causal coordinate.
- RQ3 compositional generalization.
- RQ4 summarization location.
- RQ6 fair geometry-model comparison.
- RQ7 activation–belief path faithfulness.

### Tier B: strengthens the causal story

- RQ8 safe steering range.
- RQ9 transfer.
- RQ10 uncertainty/support.
- RQ12 circuit mechanism.

### Tier C: ambitious extensions

- RQ5 full valence–arousal factorization.
- RQ11 SAE/block-sparse representation.
- RQ13 path independence/noncommutativity.
- RQ14 binding and persistence.

Do not attempt every Tier C question before the Tier A comparison is locked. A focused negative answer to RQ1–RQ7 is stronger than a large set of suggestive projections.
