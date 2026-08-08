# DRAFT — upstream issue for IINemo/lm-polygraph. NOT FILED. Review before submitting.

---

**Title:** `_SanitizeLogitsProcessor` silently converts non-finite logits into a uniform distribution, turning numerical failure into plausible-looking UQ scores

**Labels:** bug

---

### Summary

When a model produces non-finite logits, `WhiteboxModel._SanitizeLogitsProcessor`
rewrites the affected rows to **all zeros**. After `log_softmax` that is a
*uniform distribution over the vocabulary*, which is a perfectly well-formed
input to every estimator in the library. The run completes, every statistic is
finite, and the uncertainty scores look plausible — but they are computed from a
distribution the model never produced.

For a UQ library specifically, this is worse than a crash: the failure mode is
invisible in exactly the quantities a user would inspect to check for it.

### Where

`src/lm_polygraph/utils/model.py`, `WhiteboxModel._SanitizeLogitsProcessor`:

```python
row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
scores = torch.where(torch.isposinf(scores), row_max, scores)
scores = torch.where(torch.isneginf(scores), row_min, scores)
scores = torch.nan_to_num(scores, nan=0.0)
```

If an entire row is non-finite, `row_max`/`row_min` are themselves non-finite and
are replaced by `0`; `nan_to_num` then zeroes whatever remains. The row becomes
constant.

### Reproduction (no GPU, no model)

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

### How we hit it

Qwen2.5-3B-Instruct, fp16, T4, `attn_implementation="eager"`, left padding,
`batch_size=4`. **108 of 150** generations consisted solely of token id `0`,
which decodes to `!` in Qwen's vocabulary.

The trigger is upstream of lm-polygraph, in attention masking: Qwen2 eager
attention computes `attn_weights + causal_mask` with
`causal_mask = torch.finfo(dtype).min`. Left padding leaves the leading query
positions attending to nothing, so their entire row is masked. In fp16
`finfo.min` is `-65504`, and adding any ordinary negative score overflows the row
to `-inf`; `softmax` over an all-`-inf` row is `NaN`. Only the longest sequence
in each batch has no padding and therefore no fully-masked row — 1 in 4 at
`batch_size=4`, matching the 28% of samples that survived.

`transformers` guards precisely this via
`AttentionMaskConverter._unmask_unattended`, but only when
`_attn_implementation == "sdpa"` **and** `output_attentions is False`. Both
conditions fail for any lm-polygraph run using eager attention or an
attention-based estimator (`RAUQ`, `AttentionScore`, `CSL`).

**What made it hard to diagnose** is the sanitizer, not the overflow. Every
recorded diagnostic looked healthy:

* `greedy_log_probs` and `greedy_log_likelihoods` were entirely finite
* every affected sample reported a log-likelihood of exactly
  `-11.931214332580566` at every position — identical to the last digit across
  different questions, because `-ln(151936)` does not depend on the input
* the only visible symptom was `AccuracyMetric == 0`

A finiteness check on `greedy_log_probs` cannot detect this: the sanitizer runs
before those values are recorded.

### Suggested fixes

Not mutually exclusive; (1) alone would have surfaced this immediately.

1. **Warn when sanitising.** Emit a `log.warning` naming the number of rows and
   positions affected. Silence is the core problem; the repair itself is
   defensible.
2. **Do not fabricate a distribution for fully non-finite rows.** A row with no
   finite entry carries no information. Raising, or propagating `NaN` so the
   sample is dropped by the existing `_delete_nans` path in `UEManager.eval_ue`,
   is more honest than a uniform distribution that estimators will score.
3. **Record a diagnostic statistic.** A per-sample `n_sanitised_positions` in the
   stats dict would let benchmark runs report affected samples rather than
   silently averaging them in.
4. **Document the interaction** in the whitebox model docs: fp16 + left padding +
   eager attention (or `output_attentions=True`) can produce `NaN` logits,
   because transformers' guard is conditional on both.

### Environment

* lm-polygraph `efea882d810d07770e71d3a80e02416d09751435`
* transformers 4.50.0, torch 2.6.0+
* Qwen/Qwen2.5-3B-Instruct, fp16, Tesla T4 (Turing — no hardware bf16, so fp16 is
  the only half-precision option)

### Note

Found while implementing a new estimator that reads **raw** logits, where a
non-finite value cannot be normalised away and so had to be handled explicitly.
Happy to open a PR for whichever of the above you prefer.
