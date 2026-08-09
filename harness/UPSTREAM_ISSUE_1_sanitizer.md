# DRAFT 1 of 3 — NOT FILED. Review before submitting.
#
# This is the one worth filing: a genuine lm-polygraph bug, with a five-line
# reproduction needing no GPU and no model, plus a before/after from a real run.

---

**Title:** `_SanitizeLogitsProcessor` silently turns non-finite logits into a uniform distribution, so an unstable run yields plausible UQ scores instead of an error

**Labels:** bug

---

### Summary

When a model produces non-finite logits, `WhiteboxModel._SanitizeLogitsProcessor`
rewrites the affected rows to **all zeros**. After `log_softmax` that is a
*uniform distribution over the vocabulary* — a perfectly well-formed input to
every estimator in the library. The run completes, every recorded statistic is
finite, and the uncertainty scores look plausible, but they are computed from a
distribution the model never produced.

For a UQ library this is worse than a crash: the failure is invisible in exactly
the quantities a user would inspect to detect it.

### Where

`src/lm_polygraph/utils/model.py`, `_SanitizeLogitsProcessor` (line 458):

```python
row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))  # :478
row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))  # :481
scores  = torch.where(torch.isposinf(scores), row_max, scores)                      # :484
scores  = torch.where(torch.isneginf(scores), row_min, scores)                      # :485
scores  = torch.nan_to_num(scores, nan=0.0)                                         # :486
```

If an entire row is non-finite, `row_max` / `row_min` are themselves non-finite
and are replaced by `0`; `nan_to_num` zeroes the rest. The row becomes constant.

### Reproduction — no GPU, no model, five lines

```python
import torch
from lm_polygraph.utils.model import WhiteboxModel

san = WhiteboxModel._SanitizeLogitsProcessor()
V = 151936                                   # Qwen2.5 vocabulary
out = san(None, torch.full((1, V), float("nan")))

print(out.unique())                          # tensor([0.])
print(out.log_softmax(-1)[0, 0].item())      # -11.931214332580566  == -ln(V)
print(int(out.argmax()))                     # 0
```

Identical output for all-`+inf` and all-`-inf` rows.

### Observed impact

Benchmarking a new estimator on Qwen2.5-3B-Instruct, fp16, T4, TriviaQA:

| | before | after |
|---|---:|---:|
| generations that are one token repeated | **108 / 150** | 0 / 150 |
| `AccuracyMetric` | **0.0000** | 0.2930 |

"After" is the same code and data with `attn_implementation="sdpa"` and
`output_attentions=False`, which avoids the NaN source (a separate `transformers`
interaction, filed separately). Nothing about the sanitizer changed between the
two — the point is only that the corruption was real and that the sanitizer hid
it.

**What made it hard to diagnose was the sanitizer, not the NaN.** Every recorded
diagnostic looked healthy:

* `greedy_log_probs` and `greedy_log_likelihoods` were entirely finite;
* every affected sample reported a log-likelihood of exactly
  `-11.931214332580566` at *every* position, identical to the last digit across
  different questions — because `-ln(151936)` does not depend on the input;
* the only visible symptom was `AccuracyMetric == 0`.

A finiteness check on `greedy_log_probs` cannot detect this: the sanitizer runs
before those values are recorded. We initially concluded from those finite
log-likelihoods that no overflow had occurred, which was wrong — recognising
`-11.9312` as `-ln(V)` is what identified the mechanism.

### Suggested fixes

Not mutually exclusive; (1) alone would have surfaced this immediately.

1. **Warn when sanitising.** A `log.warning` naming the number of rows and
   positions affected. The silence is the problem; the repair itself is
   defensible.
2. **Do not fabricate a distribution for a fully non-finite row.** Such a row
   carries no information. Raising, or propagating `NaN` so the sample is dropped
   by the existing `_delete_nans` path in `UEManager.eval_ue`, is more honest
   than a uniform distribution that estimators will happily score.
3. **Record a diagnostic statistic.** A per-sample `n_sanitised_positions` in the
   stats dict would let benchmark runs report affected samples instead of
   silently averaging them in.

### Environment

* lm-polygraph `efea882d810d07770e71d3a80e02416d09751435`
* transformers 4.50.0, torch 2.6.0+cu124
* Qwen/Qwen2.5-3B-Instruct, fp16, Tesla T4

### Note

Found while implementing a new estimator that reads **raw** logits, where a
non-finite value cannot be normalised away and had to be handled explicitly.
Happy to open a PR for whichever option you prefer.
