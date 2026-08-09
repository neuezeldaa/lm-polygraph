# DRAFT 2 of 3 — NOT FILED. Review before submitting.
#
# Deliberately framed as DOCUMENTATION, not a bug report. The defect is in how
# transformers gates its own guard; lm-polygraph only exposes it. Filing it as a
# bug against lm-polygraph would be misattribution.

---

**Title:** Docs: attention-based estimators produce NaN logits under fp16 + left padding, because `output_attentions=True` disables the `transformers` guard

**Labels:** documentation

---

### Summary

Any lm-polygraph estimator that consumes attention weights — `RAUQ`,
`AttentionScore`, `CSL` — forces `output_attentions=True`. Under fp16 with left
padding, that silently disables a `transformers` guard and the model returns
`NaN` logits for every padded sequence in the batch.

This is not an lm-polygraph defect: the gating is in `transformers`, and
lm-polygraph only meets the conditions. But nothing warns the user, and the
downstream symptom is disguised by `_SanitizeLogitsProcessor` (filed separately),
so it seems worth a line in the whitebox docs.

### Mechanism

Qwen2 (and the other eager attention paths) compute

```python
attn_weights = attn_weights + causal_mask      # causal_mask = torch.finfo(dtype).min
```

Left padding leaves the leading query positions attending to nothing, so their
entire row is masked. In fp16 `finfo.min` is `-65504`, and adding any ordinary
score overflows the row to `-inf`; `softmax` over an all-`-inf` row is `NaN`.

`transformers` guards exactly this case with
`AttentionMaskConverter._unmask_unattended` — whose docstring names left padding
explicitly — but the call is gated:

```python
if (self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type in ["cuda", "xpu"]
        and not output_attentions):
    causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
```

An attention-based estimator violates **both** the first and the last condition.

### Reproduction — no model needed

```python
import torch
for dtype in (torch.float16, torch.float32):
    scores = torch.full((1, 8), -30.0, dtype=dtype)
    mask   = torch.full((1, 8), torch.finfo(dtype).min, dtype=dtype)
    s = scores + mask
    print(dtype, "inf:", bool(torch.isinf(s).any()),
          "nan after softmax:", bool(torch.isnan(torch.softmax(s, -1)).any()))
# torch.float16 inf: True  nan after softmax: True
# torch.float32 inf: False nan after softmax: False
```

Observed at scale on Qwen2.5-3B-Instruct, fp16, T4, `batch_size=4`, left padding:
**108 of 150** generations collapsed to token id `0`. Only the longest sequence
in each batch has no padding and therefore no fully-masked row — 1 in 4 at
`batch_size=4`, matching the 28% that survived.

### Workaround

Two configurations avoid it:

* `attn_implementation="sdpa"` **and** `output_attentions=False` — both are
  required, since the guard checks both. This is fine for estimators that do not
  read attention.
* `batch_size=1` for the attention-based estimators. With one sequence per batch
  there is no padding, so no row is ever fully masked. This is what we used, at
  the cost of throughput.

### Suggested change

A note in the whitebox model documentation, and ideally a warning when
`output_attentions=True` coincides with fp16 and a batch size above 1 — the
combination is silently wrong rather than slow.

### Environment

* lm-polygraph `efea882d810d07770e71d3a80e02416d09751435`
* transformers 4.50.0, torch 2.6.0+cu124
* Qwen/Qwen2.5-3B-Instruct, fp16, Tesla T4 (Turing: no hardware bf16, so fp16 is
  the only half-precision option)
