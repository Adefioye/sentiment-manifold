# Sentiment-geometry working conventions

Use the project skill at [`.claude/skills/sentiment-research/SKILL.md`](.claude/skills/sentiment-research/SKILL.md) for sentiment-direction, mechanistic-interpretability, dataset, intervention, evaluation, and reproduction work.

## Tigges et al. source-of-truth rule

When the user asks a question about Tigges et al., answer it based on the paper *Language Models Linearly Represent Sentiment* and the authors' local reference implementation in [`../eliciting-latent-sentiment`](../eliciting-latent-sentiment/). Consult the relevant paper material and code before answering. Clearly distinguish claims stated in the paper from behavior found only in the code, and explicitly report any discrepancy between them rather than silently reconciling it or answering from memory.

Preserve the separation among replication, discovery, and confirmation. Do not change the ToyMovieReview or SST benchmark while claiming an exact reproduction. Fit on train data, select layers and hyperparameters on validation data, and report locked test/OOD results without reselection.

Treat probes and geometry as hypotheses, not mechanisms. Require causal intervention and appropriate controls before making representation claims. Compare nonlinear methods to dimension-matched linear baselines and preserve failed results.

Keep model-specific behavior inside `sentiment_geometry/models/`, activation collection inside
`sentiment_geometry/activations/`, and fitting logic behind the common API in
`sentiment_geometry/fitting_methods/`. Dataset modules must remain free of model execution,
interventions must not calculate aggregate metrics, and experiments must be configuration-driven.
Support CUDA, MPS, and CPU; never assume a particular accelerator or Colab filesystem.

## API and software-design rules

- Name modules, classes, functions, commands, configurations, and artifact directories after the
  scientific operation or domain concept they implement. Do not expose research-question numbers,
  paper section numbers, notebook cell names, or temporary planning labels in application APIs.
- Keep command-line handlers thin. They may parse arguments and call a public package API, but
  dataset construction, fitting, evaluation, reporting, and persistence must remain independently
  usable from Python.
- Give each module one clear responsibility. Separate experiment configuration, manifest creation,
  direction fitting, orchestration, evaluation, and result summarization rather than collecting an
  entire workflow in one runner file.
- Prefer small composable functions and cohesive classes with explicit dependencies. Avoid global
  mutable state, hidden I/O, duplicate metric logic, and functions that mix loading, computation,
  plotting, and serialization.
- Use typed domain objects at subsystem boundaries. Validate invariants when objects are created and
  keep model-specific details, dataset schemas, and storage conventions behind their respective
  interfaces.
- Preserve backwards compatibility only for intentional public APIs. Do not retain a poorly named
  internal interface merely because it was recently introduced; replace it and update all callers,
  tests, documentation, configurations, and scripts together.
- Test public behavior and scientific invariants, not private implementation structure. Every new
  experiment API should support focused unit tests and a small end-to-end smoke run.
- Keep the root package organized by domain. Do not add a `src/` layout, generic `utils/` package,
  question-numbered module, or monolithic runner. Add functionality to the domain that owns it and
  depend on public APIs across domain boundaries.
