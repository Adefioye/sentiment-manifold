# Interpretability illusions, their rebuttal, and tests for learned sentiment directions

## Executive conclusion

Makelov et al. establish an important negative result: **a one-dimensional patch can control a model's output without identifying a one-dimensional feature that the unmodified model naturally uses in that way**. A learned direction can combine two geometrically different roles:

1. a **sensor** component whose coefficient changes with the source variable but which is causally disconnected downstream; and
2. an **actuator** component that can change the output but is dormant, or scarcely used, on natural inputs.

The rank-one projection used by subspace patching couples these components. End-to-end success therefore proves *controllability under that intervention*, but not by itself *natural mediation*.

Wu et al.'s response makes a complementary point: distributed representations do not generally align with the row space of the next weight matrix, and valid causal abstractions need not have a unique or privileged basis. They also correctly demand held-out interchange-intervention accuracy (IIA), a meaningful high-level causal model, and circuit-level follow-up rather than treating a large continuous logit shift as sufficient evidence. However, their central numerical counterexample appears to compare a unit direction with an **unnormalized** row-space projection. Under the intervention definition stated in their own paper, normalizing that projection changes their claimed 20% recovery to 100%. This weakens that particular counterexample, although it does not invalidate their broader points about distributed computation, non-uniqueness, or weak experimental design.

For the sentiment study, the strongest defensible present claim is:

> The trained directions are effective sentiment-control interfaces under the evaluated directional replacement intervention. Final-position DAS is especially effective, but its large effects—particularly Qwen's greater-than-200% logit-difference recovery at the final residual boundary—do not yet establish that the direction is the unique, naturally used sentiment mediator.

The decisive next step is to separate **encoding**, **causal transmission**, **natural use**, **generalization**, and **intervention validity** instead of asking a single recovery number to establish all five.

---

## 1. Makelov et al.: *Is This the Subspace You Are Looking For?*

Makelov, Lange, Geiger, and Nanda's ICLR 2024 paper asks whether successful subspace activation patching can create an *illusion of mechanistic understanding*.[^1] Their target is not ordinary full-activation patching. It is a learned low-dimensional intervention—especially Distributed Alignment Search (DAS)—where the experimenter finds a direction that makes a counterfactual output occur.

### 1.1 The basic intervention

For a unit vector $v\in\mathbb{R}^d$, a source activation $h_s$, and a base/corrupt activation $h_b$, one-dimensional interchange intervention constructs

\[
h_b^{\operatorname{patch}}
=h_b+\big(v^\top h_s-v^\top h_b\big)v
=h_b+(\Delta h^\top v)v,
\qquad \Delta h=h_s-h_b.
\]

This replaces the base activation's coordinate along $v$ with the source activation's coordinate while preserving the orthogonal complement. DAS optimizes $v$ so that this intervention induces the desired counterfactual behavior.

The tempting inference is:

\[
\text{successful patch along }v
\quad\Longrightarrow\quad
\text{the model naturally represents and uses the concept along }v.
\]

Makelov et al. show that this implication is invalid without further evidence.

### 1.2 The disconnected–dormant construction

Suppose the patched activation is the hidden activation of an MLP immediately before its down-projection $W_{\text{out}}$. Decompose the learned direction into orthogonal components

\[
v=n+r,
\qquad n\in\ker W_{\text{out}},
\qquad r\in(\ker W_{\text{out}})^\perp.
\]

Here $n$ cannot directly affect the MLP output because $W_{\text{out}}n=0$. Yet the scalar patch coefficient can depend strongly on it:

\[
\Delta h^\top v=\Delta h^\top n+\Delta h^\top r.
\]

After the down-projection, the induced change is

\[
W_{\text{out}}(h_b^{\operatorname{patch}}-h_b)
=\big(\Delta h^\top n+\Delta h^\top r\big)W_{\text{out}}r.
\]

This equation is the paper's core. The term $\Delta h^\top n$ can read out which source condition was supplied, even though $n$ itself cannot reach the output. The intervention then multiplies that source-sensitive coefficient by $r$, which *can* reach the output. Thus the projector $vv^\top$ forms an artificial bridge:

\[
\underbrace{\text{source-sensitive but disconnected}}_{n}
\quad\xrightarrow{\text{rank-one patch}}\quad
\underbrace{\text{causal but naturally dormant}}_{r}.
\]

The direction succeeds as a controller even if the unpatched network never performs this sensor-to-actuator transfer.

Two definitions support the argument:

- A **causally disconnected** subspace contains activation changes erased by all relevant downstream computation.
- A **dormant** subspace can affect the output when assigned sufficiently unusual values, but natural variation along it has little or no effect on the behavior under study.

The important distinction is between *can affect* and *does naturally mediate*.

### 1.3 IOI experiment: an apparently causal MLP direction

The first empirical case is GPT-2 Small on indirect-object identification (IOI). Prior circuit work suggests that S-inhibition heads write name-position information to the last-token residual stream, which later name-mover heads read. The authors train one-dimensional DAS directions at layer 8 in both:

- post-nonlinearity MLP activations, $v_{\mathrm{MLP}}$; and
- the residual stream, $v_{\mathrm{resid}}$.

They evaluate fractional logit-difference decrease (FLDD):

\[
\operatorname{FLDD}
=1-\frac{LD_{\mathrm{patched}}}{LD_{\mathrm{base}}},
\qquad
LD=\ell_{IO}-\ell_S.
\]

Thus 0% means no change, 100% means the original pairwise margin has been eliminated, and values above 100% mean the margin has reversed on average. They also report IIA: the fraction of examples on which the intervention produces the high-level model's desired discrete counterfactual prediction.

The central results are:

| Intervention | FLDD | IIA |
|---|---:|---:|
| Full MLP activation | -8.0% | 0.0% |
| $v_{\mathrm{MLP}}$ | 46.7% | 4.2% |
| MLP row-space component | 13.5% | 0.2% |
| MLP null-space component | 0.0% | 0.0% |
| Full residual stream | 123.6% | 54.8% |
| $v_{\mathrm{resid}}$ | 140.7% | 74.8% |
| Residual “row-space” component | 127.5% | 63.1% |
| Residual “null-space” component | 13.9% | 0.4% |
| Downstream-gradient direction | 111.5% | 45.1% |

The MLP result has the suspicious signature predicted by the toy construction:

1. patching the full MLP activation does essentially nothing;
2. patching a learned one-dimensional direction has a much larger graded effect;
3. deleting the direction's null-space component greatly reduces that effect; and
4. the null-space component alone has no effect.

The last two statements are not contradictory. The null-space component supplies the *coefficient* that determines how much of the row-space component the rank-one projector injects. It is causally inert alone but causally important inside the compound projector.

The authors additionally show that after patching, the MLP suddenly writes much more strongly in a direction that affects the name-mover heads. This supports the claim that the intervention activates a pathway weakly used on natural examples. They reproduce a similar effect using randomly sampled MLP weights, although that experiment adds the random MLP's patch-induced output difference to the real network rather than literally showing the original network naturally uses a random circuit.

### 1.4 The faithful comparison case: the residual-stream bottleneck

The paper does not conclude that DAS or subspace patching is generally invalid. It presents the residual-stream direction as a comparatively faithful success case.

Three facts matter:

1. The residual stream at the selected location is a computational bottleneck: earlier components cannot bypass the intervention through a parallel residual route in the way that they can bypass an individual MLP.
2. Removing the component disconnected from the downstream name-mover query matrices retains most of the causal effect.
3. Independent circuit evidence identifies plausible writers (S-inhibition heads), natural separation by name position, and plausible readers (name-mover heads).

This triangulation connects a learned direction to a naturally operating pathway. The lesson is not merely “patch residual streams.” It is: favor bottlenecks and supply evidence for **natural writers, natural variation, and natural readers**.

### 1.5 Factual recall and rank-one editing

The second case fits one DAS direction per fact pair in GPT-2 XL, patching the last subject-token MLP activation so that a prompt for one subject produces the other subject's object. Across 40 fact pairs from eight relations, learned directions often change the recalled object at layers where:

- full MLP patching has negligible effect; and
- row-space-only patching loses the learned direction's effect.

The subject-activation difference is reported to align strongly with the null component (about 0.9 cosine) and more weakly with the causal row component (about 0.4), again matching the sensor–actuator account.

They then prove that a rank-one activation patch at one token has an equivalent rank-one edit to $W_{\text{out}}$ at that token. If

\[
\delta y=(\Delta h^\top v)W_{\text{out}}v,
\]

one can choose a rank-one update $W'_{\text{out}}=W_{\text{out}}+ab^\top$ whose action on the selected hidden state produces the same $\delta y$. This provides a mechanistic explanation for why a rank-one fact edit may work at a layer without proving that the fact was naturally *stored* there: editing demonstrates a place where an effective control can be inserted, not necessarily the original storage location.

### 1.6 Why the authors expect the problem to be common

Transformer MLP hidden dimensions are usually wider than the residual stream, so the down-projection has a large kernel. Residual connections also allow causal signals to bypass individual MLPs. If a feature is already recoverable upstream, an intervening but task-irrelevant MLP can contain:

- a null-space direction that preserves/re-encodes the upstream feature; and
- a row-space direction capable of writing a behaviorally effective residual signal.

Optimization can combine them. Sparse circuits, wide MLPs, and skip connections therefore make the construction structurally plausible, though the paper does not prove a prevalence rate across tasks and models.

### 1.7 What to learn from which pages

| PDF pages | What they contribute |
|---|---|
| 2–3 | Thesis, visual intuition, and contribution summary. |
| 4–5 | Formal disconnected/dormant construction and minimal example. |
| 5–7 | IOI setup, DAS directions, FLDD/IIA, and evidence for the MLP illusion. |
| 7–8 | Residual-stream success case and why a bottleneck changes the argument. |
| 8–9 | Factual recall, equivalence to rank-one editing, prevalence argument, recommendations. |
| 17 | Random-MLP validation and its exact experimental construction. |
| 19–20 | Factual-recall decompositions and proof of rank-one-edit equivalence. |
| 21 | Empirical argument that MLP nonlinearities need not destroy upstream linear information. |

---

## 2. Wu et al.: the response sometimes called “Illusions of Interpretability Illusions”

The paper's actual title is *A Reply to Makelov et al. (2023)'s “Interpretability Illusion” Arguments*.[^2] It is a 2024 arXiv preprint; I did not find evidence that this reply itself was published at a peer-reviewed venue. The informal “illusion of illusions” description captures its rhetorical aim, but should not be used as the bibliographic title.

### 2.1 Central thesis

Wu, Geiger, Huang, Arora, Icard, Potts, and Goodman reject Makelov et al.'s label more than they reject the underlying geometry. Their position is:

- distributed representations naturally have components in downstream null spaces;
- the upstream data manifold need not be orthogonal to a downstream kernel;
- a representation can therefore look “illusory” under Makelov's diagnostic while still supporting an intuitive causal abstraction; and
- multiple valid causal abstractions can coexist because neural representations have no universally preferred basis.

They reinterpret Makelov's phenomenon as a normal property of distributed representation rather than automatically an experimental artifact.

### 2.2 Their algebraic restatement

Using $v=n+r$, they expand the downstream effect as

\[
\Delta y_v
=\big(\Delta h^\top n+\Delta h^\top r\big)W_{\text{out}}r.
\]

They define an “illusion effect” as the difference between downstream behavior after patching along the full $v$ and after patching along its row-space component $r$. Their objection is that this difference can be nonzero even when $v$ intuitively identifies a real, naturally used variable.

### 2.3 The toy counterexample

They consider a linear network computing the identity:

\[
h(x)=W_1x=[x,0,x]^\top,
\qquad
f(x)=W_2^\top h(x),
\qquad
W_2=[0,2,1]^\top,
\]

so $f(x)=x$. The third hidden unit $e_3=[0,0,1]$ is presented as the obvious faithful mediator. Yet relative to $W_2$,

\[
e_3=n+r,
\quad n=[0,-0.4,0.8],
\quad r=[0,0.4,0.2].
\]

Their displayed calculation patches with $r$ directly and obtains only 20% of the full intervention effect. Therefore, they argue, Makelov's row-space-removal diagnostic wrongly labels the obvious third-unit explanation as illusory.

This example is rhetorically central, but it contains a normalization problem discussed in Section 3.2 below.

### 2.4 Critique of Makelov's empirical evidence

Wu et al. make three important experimental criticisms.

First, they prioritize **interchange-intervention accuracy**. IIA measures whether the low-level network under an intervention matches the counterfactual prediction of a specified high-level causal model:

\[
\operatorname{IIA}
=\Pr\big[F_{\text{low}}^{\operatorname{do}(Z\leftarrow z')}(x)
=F_{\text{high}}^{\operatorname{do}(Z\leftarrow z')}(x)\big].
\]

They argue that the MLP direction's 4.2% IIA is weak evidence when the model solves IOI at roughly 92–96% accuracy. A 46.7% average FLDD can arise from subthreshold margin changes that rarely produce the intended counterfactual behavior.

Second, correlational activation separation does not prove causal use. The finding that a null-space projection separates conditions more strongly than a row-space projection is evidence of encoding, not natural mediation.

Third, the factual-recall design trains one direction on one fact pair and evaluates it on that same pair. A flexible optimization method can learn a “return the source answer” controller without learning a general fact variable. They recommend many training examples, disjoint testing examples, and a substantive high-level causal model.

### 2.5 Their positive IOI experiments

The reply then uses DAS as a circuit-discovery tool under a more explicit causal model and held-out design. Their results include:

- Name-position information reaches about 70% IIA in the layer-8 last-token residual stream but only about 4% in MLP activations.
- Earlier single-token positions, and a concatenation of earlier positions, have near-zero IIA for that variable.
- Within layer 8, IIA rises from about 11% at block input to 48% at attention output and 69% at block output, suggesting attention writes the name-position signal.
- No individual head yields good IIA; leave-one-out and cumulative-head tests indicate a distributed contribution, with heads 6 and 10 particularly influential.
- The correct IO-name variable emerges in later attention computations. Vanilla interchange intervention obtains about 85% IIA at the relevant output locations, while DAS finds additional lower-level alignments, including about 13% in a value stream where vanilla patching gives 0%.

These results support a nuanced claim: DAS can identify distributed causal structure when the high-level variable, intervention semantics, train/test split, and surrounding circuit are well specified.

### 2.6 What to learn from which pages

| PDF pages | What they contribute |
|---|---|
| 2–3 | Reply's thesis, Makelov formalism, and null/row decomposition. |
| 4–5 | Toy identity network, proposed counterexample, data-manifold and multiple-abstraction arguments. |
| 5–7 | IIA versus FLDD, warning about correlational evidence, and factual-recall overfitting critique. |
| 8–9 | Held-out IOI setup; last-token residual versus MLP and earlier-position findings. |
| 9–12 | Attention-stream and multi-head localization; emergence of the IO-name variable. |
| 12 | Overall conclusion. |

---

## 3. Independent algebraic critique

### 3.1 The papers answer different causal questions

Much of the dispute dissolves after distinguishing two estimands:

1. **Causal abstraction/control:** Does replacing a learned variable make the neural network agree with the counterfactual behavior of a high-level model on a specified intervention set?
2. **Natural mediation/mechanism:** Does the unmodified network normally transmit the relevant information through this variable, along this route, on natural inputs?

High held-out IIA is strong evidence for the first claim. It does not logically entail the second. Conversely, a representation can participate in a distributed abstraction even if an individual component of its coordinate vector lies in the kernel of the immediately following linear map.

Makelov is strongest about the gap between (1) and (2). Wu is strongest about not imposing a unique coordinate system on (1). The sentiment project should report the two claims separately.

### 3.2 A normalization inconsistency in Wu et al.'s toy example

Both papers define a one-dimensional patch using a unit direction (or, equivalently, an orthonormal basis). Wu et al.'s projected row vector is

\[
r=[0,0.4,0.2],
\qquad \|r\|^2=0.2.
\]

They insert this unnormalized vector into the rank-one expression $rr^\top$. This is not the orthogonal projector onto $\operatorname{span}(r)$; the projector is

\[
P_r=\frac{rr^\top}{r^\top r}
=\hat r\hat r^\top,
\qquad \hat r=\frac{r}{\sqrt{0.2}}.
\]

For their example with $x=1$, $x'=5$,

\[
\Delta h=[4,0,4].
\]

The correctly normalized row-space patch is

\[
(\Delta h^\top\hat r)\hat r=[0,1.6,0.8].
\]

Applying $W_2^\top=[0,2,1]$ gives

\[
W_2^\top[0,1.6,0.8]=4,
\]

which changes the output from 1 to 5: **100% recovery**, not 20%. The published footnote claims normalization makes the “illusion” larger, but the stated numbers yield the opposite conclusion.

This is an independent algebraic check of the displayed example, not a claim about every experiment in the reply or in Makelov et al. It means that this particular toy example does not establish what the authors say it establishes under their stated unit-vector intervention.

More generally, if $v=n+r$ is unit and the row-only patch uses $\hat r=r/\|r\|$, then

\[
\Delta y_{\hat r}
=\frac{\Delta h^\top r}{\|r\|^2}W_{\text{out}}r,
\]

not $(\Delta h^\top r)W_{\text{out}}r$. Comparisons of full and projected directions must therefore state whether they preserve direction, projector, coefficient scale, or downstream effect. These are different interventions.

### 3.3 Makelov's diagnostic is revealing but not a universal definition of illusion

The immediate $W_{\text{out}}$ kernel is a useful structural diagnostic at post-MLP activations because it exactly identifies changes erased by that linear map. But several qualifications matter:

- “Dormant” is distribution-relative. A direction dormant on one prompt family may be naturally active under negation, sarcasm, longer contexts, or another downstream task.
- A row/null decomposition is component-relative. The residual stream, attention outputs, and MLP hidden states have different downstream maps and bypass structures.
- A large full-versus-row difference is evidence for cross-coupling, but the normalization and calibration of the compared interventions matter.
- Failure of one local mechanism hypothesis does not mean the direction is useless. It may still be a robust control interface or a member of an equivalence class of causal abstractions.

Thus Makelov supplies a strong *falsification test* for a naive mechanistic story, not a complete necessary-and-sufficient definition of representation.

### 3.4 Wu's “distributed geometry” does not by itself establish natural use

Wu et al. are right that the data manifold need not be orthogonal to a downstream null space. But this fact does not answer whether the **specific cross-term created by the patch** occurs in the natural forward computation. Algebraically,

\[
(\Delta h^\top n)W_{\text{out}}r
\]

is introduced by the rank-one intervention. The unmodified layer computes $W_{\text{out}}h$, not $vv^\top\Delta h$ followed by $W_{\text{out}}$. Natural null-space variation is therefore compatible with Makelov's central warning: an intervention may use that variation as a control signal in a way the model does not.

The reply establishes that null-space alignment alone cannot settle faithfulness. It does not establish that a high-performing learned patch necessarily traces the naturally used path.

### 3.5 Euclidean subspaces are not invariant to general reparameterization

Both approaches inherit a deeper ambiguity. Under an invertible change of coordinates $h'=Ah$, the model can be reparameterized to compute exactly the same function. Yet the Euclidean projector generally does not transform covariantly:

\[
P_{Av}\neq AP_vA^{-1}
\]

unless $A$ is orthogonal (with appropriate normalization). Consequently, “the one-dimensional direction” and its orthogonal complement depend on the chosen activation metric and basis.

This does not make directional analysis meaningless—the trained network provides a concrete parameterization—but it weakens claims of a uniquely privileged direction. Whitening by the natural activation covariance, using a local Jacobian/Fisher metric, and testing stability under function-preserving reparameterizations would make the geometric claim more principled.

### 3.6 Separate sensing from actuation explicitly

Ordinary directional replacement forces one vector to play two roles:

\[
\delta h=(\Delta h^\top d)d.
\]

A more diagnostic family uses an independently chosen sensor $s$ and actuator $a$:

\[
\delta h=(\Delta h^\top s)a.
\]

This permits four targeted interventions:

| Sensor | Actuator | Question |
|---|---|---|
| $d$ | $d$ | Does the original learned patch work? |
| row/causal component | same row component | Does a self-contained causal component work? |
| null/disconnected component | row/causal component | Is the effect specifically a sensor–actuator cross-term? |
| natural sentiment probe | output-gradient direction | Can arbitrary encoding be converted into output control? |

Calibration must be learned on training data and frozen before evaluation so that vector norms cannot decide the comparison.

### 3.7 FLDD, IIA, KL, and overshoot measure different things

Wu is correct that a continuous logit metric can be large while few decisions flip. But IIA also discards meaningful graded changes and saturates once the desired token wins. Neither should replace the other.

For sentiment, report at least:

- raw clean, corrupt, and patched logit differences;
- normalized logit-difference recovery, with confidence intervals;
- sign-flip/counterfactual accuracy;
- $D_{KL}(p_{\text{clean}}\|p_{\text{patched}})$ and its improvement relative to corrupt;
- changes in the individual positive and negative verbalizer logits;
- generated continuations or class probabilities under several verbalizer sets.

A result above 100% is not intrinsically invalid. It means the patched pairwise margin lies beyond the clean margin along that scalar metric. It can reflect removal of an opposing orthogonal contribution, nonlinear amplification, repeated all-token interventions, or an optimized readout shortcut. It is a diagnostic to decompose, not a score to celebrate unqualifiedly.

### 3.8 Later work strengthens the middle position

Three later lines of work make a synthesis more compelling:

- Grant et al. distinguish harmless null-space divergence from harmful “hidden pathway” divergence and regularize learned interventions toward natural counterfactual activations.[^3]
- Mueller shows that causal abstraction can be ambiguous under overdetermination and non-transitive counterfactual criteria; one intervention score need not identify a unique causal explanation.[^4]
- Vaidyanathan et al. rederive activation patching as a mediation estimand containing interaction terms. Patch effects can be inflated or hidden when a component's effect depends on the state of other components; these terms grow with intervention distance and disappear in locally affine regimes.[^5]

These works support neither “all successful patches are illusions” nor “successful held-out IIA settles mechanism.” They support measuring intervention distance, interactions, generalization, and pathway consistency.

---

## 4. What this means for the trained sentiment directions

### 4.1 Immediate interpretation of the current results

The current experiment learns mean-difference, logistic-regression, and one-dimensional DAS sentiment directions from adjective or final-token activations, then applies directional replacement across token positions and layers. The striking cases include approximately:

- GPT-2 Small final-token DAS: 157% recovery on toy adjectives and 166% on toy verbs;
- Qwen3-0.6B final-token DAS: 216% on toy adjectives and 235% on toy verbs;
- materially lower but still strong OOD recovery on adverbs and SST.

Several details sharpen the interpretation:

1. The Qwen final boundary is after the last transformer block and before final normalization/readout. At the last token, this is effectively a terminal decision-state intervention. A direction there can be an excellent sentiment controller without identifying where sentiment was computed.
2. Directional replacement is performed at all positions in the evaluation. Repeated injections can accumulate and interact.
3. DAS is trained directly on normalized logit-difference recovery. With clean margin greater than corrupt margin, the implemented loss is minimized at clean recovery but becomes *negative* when the patch overshoots. The training objective therefore rewards overshoot unless separately constrained.
4. Best layers are selected independently for each method, position, dataset, and metric. This creates winner's-curse bias, especially for Qwen with more candidate layer boundaries.
5. Some toy OOD sets are tiny (for example, only four directed Qwen verb cases), so extreme percentages can be dominated by a few denominators or examples.

These considerations do not explain away the result. They define the alternative hypotheses the next study must distinguish.

### 4.2 Competing hypotheses

**H1 — Natural residual sentiment mediator.** The direction represents a sentiment variable naturally written by upstream components and read by downstream components. It should generalize across lexical category, template, negation, intensity, and verbalizer; work bidirectionally; show plausible writers/readers; and remain effective under on-manifold, earlier-layer interventions.

**H2 — Terminal readout controller.** Final-position DAS primarily learns a direction aligned with the local gradient or unembedding contrast for the selected positive/negative output tokens. It will be strongest at the last token/final boundary, depend on the verbalizer, and show weaker transport to earlier layers or positions.

**H3 — Sensor–actuator coupling.** One component separates positive from negative source activations but is weakly causal, while another low-variance component controls the output. The rank-one patch connects them. A factorized null-sensor/causal-actuator intervention will reproduce much of the effect, while full activation patching and the self-contained causal component will not.

**H4 — Cancellation or interaction overshoot.** In the clean run, the learned direction contributes more positive margin than the final total because the orthogonal complement contributes a negative offset. Replacing only the direction removes this cancellation and exceeds 100%. Alternatively, joint token or nonlinear layer interactions amplify the patch.

**H5 — A non-unique causal abstraction.** Many geometrically different directions implement equally good held-out counterfactual control. Sentiment is causally controllable through an equivalence class rather than uniquely localized to one axis.

---

## 5. Targeted research questions and decisive tests

| Research question | Experiment | Evidence favoring a naturally used sentiment direction |
|---|---|---|
| **RQ1. Does the same subspace encode and causally transmit sentiment?** | Decompose each direction relative to the immediate downstream Jacobian or component map. Run normalized/calibrated full, causal-component-only, disconnected-component-only, and disconnected-sensor→causal-actuator interventions. | The causal component both separates sentiment naturally and retains most held-out effect; the cross-term is not the dominant source of success. |
| **RQ2. Is final DAS a verbalizer/readout shortcut?** | Compare its cosine with the local logit-difference gradient, unembedding-vector differences, and RMSNorm-aware Jacobian directions. Re-evaluate with multiple unseen label words and label-free generation measures. | Effects survive new verbalizers and are not explained by near-collinearity with the selected output contrast. |
| **RQ3. Does the direction travel through the model?** | Fit once, patch one position at a time and at earlier residual boundaries; exclude the last token; use path patching from hypothesized writers to readers. | A coherent layer/position trajectory appears before the terminal readout, with identifiable components writing and reading the direction. |
| **RQ4. Why is recovery above 100%?** | Record clean/corrupt/patched raw margins per example. At the final boundary, decompose logit difference into contribution along $d$, its complement, and normalization-induced interaction. Compare full-state versus directional patches. | Overshoot is reproducible with fixed layers and large samples and admits a stable mechanistic decomposition rather than being denominator noise or direct objective exploitation. |
| **RQ5. Is the patch on the natural counterfactual manifold?** | Measure nearest-neighbor distance, Mahalanobis distance, local-PCA residual, and covariance-whitened distance from patched states to real target-class states. Project patches onto a target local manifold or add counterfactual-locality regularization. | Causal performance remains high at natural-sized intervention distances and after locality constraints. |
| **RQ6. Does the direction express the intended high-level variable?** | Train on adjectives; freeze method/layer/hyperparameters; test unseen lexemes, verbs, adverbs, SST, negation, intensifiers, contrastive clauses, mixed sentiment, and sarcasm. | Effects track compositional sentence sentiment, not merely lexical polarity or prompt format. |
| **RQ7. Is it necessary as well as sufficient?** | Test denoising and noising, both positive→negative and negative→positive, graded erasure, and concept ablation. | Removing the direction selectively harms sentiment behavior, while restoring it repairs behavior in both directions. |
| **RQ8. Is the direction unique or an equivalence class?** | Train many DAS seeds and dimensions; compare principal angles, probe accuracy, IIA, and cross-patching. Search explicitly for mutually orthogonal high-IIA directions. | Successful runs converge geometrically and share circuit readers, or the paper explicitly reports non-identifiability rather than a unique axis. |
| **RQ9. Do token/component interactions inflate attribution?** | Compare each single-position patch, the sum of their individual effects, and the simultaneous all-position patch. Compute $I=\Delta_{\mathrm{all}}-\sum_t\Delta_t$; repeat for component groups and smaller interpolation steps. | The conclusion survives when interaction terms are small or are explicitly modeled and stable. |
| **RQ10. Is DAS better because it finds mechanism or because it optimizes the metric?** | Evaluate all directions on objectives not used for training; train DAS on IIA or KL rather than recovery; test mean/LR/DAS with equal norm, layer-selection budget, and held-out validation. | DAS superiority persists across metrics, fixed layers, seeds, and an untouched test set. |
| **RQ11. Is one dimension sufficient?** | Compare 1D DAS with 2–8D and Boundless DAS; test graded valence and mixed/aspect sentiment. | One dimension reaches the multidimensional ceiling without systematic failures on compositional cases. |
| **RQ12. Are effects specific beyond generic high-gain directions?** | Use norm-, variance-, sparsity-, and gradient-alignment-matched random controls; permuted sentiment labels; random source pairing; and random-MLP-style controls. | The learned direction exceeds the full matched null distribution and remains semantically selective. |

### 5.1 A high-value experimental sequence

The following order will discriminate the hypotheses efficiently.

1. **Freeze selection.** Choose layers and all hyperparameters using only training/validation data. Reserve a genuinely untouched test set. Increase the tiny directed verb samples.
2. **Make the metric panel.** For every example save raw logits, recovery, IIA/sign flip, clean-referenced KL, individual label-logit changes, and generated output. Report bootstrap intervals and denominator distributions.
3. **Split positions.** Re-run single-position patches, especially “all except final token” and “final token only.” This directly tests terminal control versus transported representation.
4. **Run the final-boundary identity check.** At the pre-readout residual state, compare full clean-state replacement with directional replacement. If a direction exceeds the full patch, decompose the clean state into direction/complement contributions through final normalization and unembedding.
5. **Test readout dependence.** Rotate verbalizers and prompts; measure gradient and unembedding alignment. A semantic direction should not collapse when “positive/negative” output tokens change.
6. **Factor sensor and actuator.** Use the local downstream Jacobian to form low-effect and high-effect components, normalize projectors correctly, calibrate on training data, and test all sensor→actuator combinations on held-out data.
7. **Constrain locality.** Interpolate $\alpha\in[-2,2]$, mark the natural source-coefficient range, calculate manifold distances, and test locality-regularized DAS.
8. **Trace the circuit.** Attribute which attention heads/MLPs write the direction and which later modules read it. Confirm with path-specific patching or causal scrubbing.
9. **Test non-uniqueness.** Repeat many seeds and search for orthogonal alternatives with comparable held-out causal scores.

### 5.2 Criteria for a firm causal-faithfulness claim

A strong conclusion would require all of the following:

- **Generalization:** the frozen direction works on unseen lexical items, constructions, datasets, and verbalizers.
- **Counterfactual correctness:** high IIA/sign-flip accuracy accompanies continuous margin recovery.
- **Bidirectionality and selectivity:** both sentiment directions work, and unrelated behaviors remain comparatively stable.
- **Natural-range interventions:** successful coefficients resemble target-class coefficients and patched states remain near the natural counterfactual manifold.
- **Mechanistic continuity:** identifiable upstream components write the signal and downstream components read it.
- **No dominant patch-only bridge:** a disconnected-sensor→actuator cross-term does not explain most of the effect.
- **Metric robustness:** conclusions survive raw logit difference, KL, probability, and generation-based evaluations.
- **Interaction accounting:** conclusions remain under single-position/component patches or explicitly include stable interaction terms.
- **Selection honesty:** layers and thresholds are selected without touching the reported test set.
- **Non-uniqueness disclosure:** stability across seeds is demonstrated, or the result is framed as an equivalence class of control directions.

If only held-out causal control and semantic generalization hold, the appropriate claim is still valuable but narrower: **a versatile sentiment intervention subspace**. If writers/readers, natural-range locality, necessity, and pathway continuity also hold, then the stronger phrase **naturally used causal sentiment representation** becomes justified.

---

## Sources

[^1]: Aleksandar Makelov, Georg Lange, Atticus Geiger, and Neel Nanda, “Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching,” ICLR 2024. [Conference page](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html); [PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/70b8505ac79e3e131756f793cd80eb8d-Paper-Conference.pdf).
[^2]: Zhengxuan Wu et al., “A Reply to Makelov et al. (2023)'s ‘Interpretability Illusion’ Arguments,” arXiv:2401.12631, 2024. [Abstract and PDF](https://arxiv.org/abs/2401.12631).
[^3]: Angus Grant et al., “Addressing Divergent Representations from Interchange Interventions,” ICLR 2026. [arXiv](https://arxiv.org/abs/2511.04638).
[^4]: Aaron Mueller, “Missed Causes and Ambiguous Effects: Counterfactuals Pose Challenges for Interpreting Neural Networks,” ICML 2024. [PMLR paper](https://proceedings.mlr.press/v235/mueller24a.html).
[^5]: Sankaran Vaidyanathan et al., “The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching,” arXiv:2606.27510, 2026. [Abstract and PDF](https://arxiv.org/abs/2606.27510).
