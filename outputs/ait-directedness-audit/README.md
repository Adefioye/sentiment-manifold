# AIT object-directedness audit

This is a discovery analysis of the 1,859 non-neutral English V-oc rows in the cached, pinned
`kokolamba/sentiment-manifold-ait-valence-binary` artifact at revision
`c53df7c117c2f433df904cfaeddb9062027f755f`. Object-directedness was not annotated by the
SemEval authors, so these labels must not be described as SemEval ground truth.

## Exclusive annotation codebook

- `target_directed`: the tweeter's affect or evaluation has an identifiable target or cause,
  including a person, self, object, event, situation, proposition, topic, or addressee.
- `objectless_general`: the tweeter expresses a mood, feeling, bodily affective state, or diffuse
  positivity/negativity without an identifiable target or cause.
- `reported_quoted`: the main affect is attributed to another person or merely reported/quoted,
  with no clear affective stance from the tweeter.
- `mixed`: substantial content from more than one of the preceding frames, with no single frame
  dominating.
- `ambiguous`: the affect holder or target cannot be resolved from the tweet.

If the tweeter clearly evaluates or reacts to a target, `target_directed` takes precedence over
`reported_quoted`. Self-directed evaluations count as target-directed.

## Close-reading sample

`manual_stratified_sample100.csv` contains a reproducible stratified sample: 50 negative rows
sampled with seed `20260929`, 50 positive rows sampled with seed `20260930`, then shuffled with seed
`20260931`. The observed counts are:

| Category | Count |
|---|---:|
| target-directed | 87 |
| objectless/general | 7 |
| reported/quoted | 3 |
| mixed | 0 |
| ambiguous | 3 |

Weighting the two strata by their corpus frequencies (994 negative, 865 positive) gives estimated
proportions of 86.65%, 7.35%, 2.93%, 0.00%, and 3.07%, respectively. The target-directed bootstrap
95% interval is approximately 79.58%–92.79%. Zero observed mixed cases does not demonstrate that the
corpus contains none; the rule-of-three upper bound is approximately 3.7% for this sample size.

## Automatic-label robustness check

`annotate_cached_ait.py` implements a constrained-label local-model audit. The 1.5B and 3B pilot
labels were highly sensitive to prompt wording and systematically overused different categories.
The generated automatic-label CSVs are retained only as failed exploratory diagnostics and must not
be used as corpus estimates. A reportable full-corpus census requires independent human annotation
with agreement statistics and adjudication.
