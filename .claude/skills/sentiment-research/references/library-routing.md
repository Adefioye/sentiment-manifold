# Mechanistic-interpretability library routing

| Need | Preferred local choice | Guardrail |
|---|---|---|
| Exact historical GPT-2 reproduction | `../eliciting-latent-sentiment` and its pinned TransformerLens fork | Do not mix modern hook semantics into a numerical replication claim. |
| Shared GPT-2/Qwen experiment | `sentiment_manifold.models.CausalLMAdapter` on Hugging Face Transformers | Keep architecture branching inside the adapter. |
| Quick cache, logit-lens, and component exploration | Modern TransformerLens / TransformerBridge | Validate hook-name and layer-boundary equivalence before comparing historical numbers. |
| General PyTorch/Hugging Face interventions | NNsight | Prefer when model coverage or remote execution matters more than exact legacy parity. |
| Interchange interventions and trained subspaces | PyVene through Causalab | Use for DAS/DBM and serializable intervention graphs; preserve task-level counterfactual semantics. |
| End-to-end causal abstraction | `../causalab` | Use baseline → locate → subspace → activation_manifold → path_steering. |
| SAE analysis | `../sae-manifold`, optionally SAE Lens | Compare feature clusters with PCA/random bases and test causal sufficiency. |
| Local affine geometry | `../decomposing-activations-local-geometry` | Select MFA components/factor dimension on held-out likelihood plus causal validation. |
| Output-belief geometry | `../shape-of-beliefs` | Use full distributions and Hellinger geometry; retain other vocabulary mass. |
| Feature blocks | `../block-sparse-featurizer` | Treat language transfer as exploratory until validated. |

Do not add a second hooking framework to a single benchmark path without an explicit parity test. Record package versions and model/tokenizer revisions in every run.

