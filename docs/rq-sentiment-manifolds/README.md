# Sentiment manifolds as causal representations

## Research objective

Determine whether sentiment is represented by a nonlinear activation manifold that is **causally more useful** than the one-dimensional sentiment direction of Tigges et al.—first in their own models, prompts, datasets, token positions, and intervention setting, and only then in richer affective and naturalistic settings.

“More useful” is deliberately stronger than “more accurately decoded” or “looks curved in a projection.” A manifold earns the claim only if it improves held-out causal behavior while controlling for dimensionality, intervention magnitude, density/support, training data, and hyperparameter budget.

The primary comparison is:

> Tigges 1D DAS direction vs. a sentiment manifold, using matched directional/interchange interventions on ToyMovieReview and Stanford Sentiment Treebank (SST), with Tigges's causal logit-difference recovery and logit-flip rate as endpoint metrics.

The strongest publishable result would be a three-part finding:

1. a manifold beats 1D DAS on the original Tigges endpoint intervention task;
2. it explains *why* through graded, neutral, mixed, or compositional sentiment states that a line conflates;
3. the advantage survives held-out lexical items, templates, domains, layers, and models without causing off-manifold degeneration.

A valid negative result is also valuable: if well-controlled manifold models do not beat 1D DAS, the conclusion should be that sentiment's nonlinear descriptive geometry is not causally useful in these settings.

## Documents

- [Literature and repository synthesis](01-literature-and-repositories.md): verified claims from the four papers, links, code availability, and the exact ingredients in every relevant workspace repository.
- [Research questions and hypotheses](02-research-questions.md): one primary and twelve secondary questions, each tied to a causal contrast, prediction, and falsifier.
- [Experimental methodology](03-methodology.md): exact Tigges replication, dataset design, representation candidates, interventions, metrics, statistical tests, and the causalab implementation map.
- [Diagnostics and recovery playbook](04-diagnostics-and-recovery.md): how to tell genuine manifolds from projection artifacts or confounds, and how to repair failed representations without moving the goalposts.
- [Execution roadmap](05-execution-roadmap.md): staged experiments, gates, artifacts, compute-conscious ordering, and a preregistration-ready definition of “beats Tigges.”

## Claim ladder

The project should not skip levels in this ladder:

| Level | Claim | Minimum evidence |
|---|---|---|
|---:|---|---|
| 0 | The model performs the task | Strong zero-shot baseline and valid counterfactual pairs |
| 1 | Sentiment is decodable | Held-out probe/CVP performance |
| 2 | Activations have nonlinear geometry | Held-out geometric evidence beyond a dimension-matched linear model |
| 3 | The geometry is causally faithful | Tangent/manifold-coordinate interventions recover intended counterfactual behavior |
| 4 | The manifold is more useful than a line | Paired superiority over 1D DAS and dimension-matched linear controls |
| 5 | The advantage generalizes | Lexical, template, domain, layer, model, and composition transfer |
| 6 | A mechanism explains the advantage | Heads/MLPs write, transport, or read manifold coordinates in a causally validated circuit |

The headline claim requires at least Level 4. Level 2 alone is not evidence against Tigges.

## Primary success criterion

On a locked test set in the Tigges setting, a manifold method must satisfy all of the following:

- **Endpoint efficacy:** higher paired counterfactual logit-difference recovery than 1D DAS, with a 95% confidence interval above zero; or non-inferior endpoint recovery together with a prespecified, substantial gain in path naturalness and specificity.
- **Binary decision efficacy:** no reduction in logit-flip rate beyond the non-inferiority margin.
- **Specificity:** no worse change to matched non-sentiment logits, perplexity, syntax/coherence, and content preservation.
- **Fairness:** the same layer, token position, train/validation/test split, number of labels, model access, and intervention-norm budget; comparisons against a dimension-matched linear subspace and a straight path in the *same* learned subspace.
- **Support:** intermediate intervention states remain in regions supported by natural activations, using held-out density or reconstruction diagnostics.
- **Robustness:** the conclusion holds across seeds and at least one lexical/template OOD split; the test set is not used to select manifold type, dimension, smoothness, layer, or step size.

If endpoint performance is already saturated, use a preregistered lexicographic rule: first require endpoint non-inferiority, then test path coherence/support, then OOD/compositional transfer. Do not quietly replace Tigges's outcome with a friendlier manifold metric.

## Central experimental contrast

For a source activation \(h_s\), target state \(t\), and representation map \(f\):

- **Tigges/DAS direction:** replace or patch only \(\langle h_s,d\rangle\) using a one-dimensional direction \(d\), preserving the orthogonal residual.
- **Linear subspace:** replace the coordinates in a learned \(k\)-dimensional affine subspace.
- **Manifold coordinate intervention:** project to or infer intrinsic coordinates \(z_s\), move to the target coordinate \(z_t\) along a learned supported path, decode to \(\hat h(z_t)\), and preserve the residual normal component only when doing so is justified and matched across methods.
- **Raw activation control:** patch the full target activation or centroid as a positive upper-bound control.
- **Off-manifold control:** make an equal-norm displacement normal to the learned tangent space.

The curvature claim is identified by comparing the manifold path to the straight chord inside the same subspace—not by comparing a high-capacity manifold to a single vector alone.

## Recommended scope

The first paper should remain about **sentiment**, not all emotion geometry. Use valence as the primary variable, then add neutral/mixedness and arousal only when they diagnose a failure of the line. GoEmotions and Claude-style behavioral tasks are extensions after the Tigges benchmark is won or decisively characterized.

The most defensible initial models are Pythia-1.4B and Pythia-2.8B because they reproduce the baseline and are inexpensive enough for causal sweeps. A current open model can be added later for external validity. Claude Sonnet 4.5 results should motivate behavioral tests, not serve as a reproducible baseline, because its activations and experiment code are not publicly available.

## Research integrity rule

Freeze the primary outcome, test split, superiority/non-inferiority margins, and model-selection protocol before opening final results. Report linear wins, ties, and manifold failures. A representation that requires test-set tuning, leaves the natural activation support, or improves only a visualization is not a successful sentiment manifold.
