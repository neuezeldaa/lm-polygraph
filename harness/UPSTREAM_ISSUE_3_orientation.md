# DRAFT 3 of 3 — NOT FILED. Review before submitting.
#
# Separate from the other two: different subsystem, and the evidence is of a
# different kind (synthetic distributions + a negation check, not a crash).
# Deliberately phrased as a question -- the intended convention is theirs to say.

---

**Title:** `RenyiNeg`, `FisherRao` and `MeanConditionalPointwiseMutualInformation` appear oriented opposite to their docstrings

**Labels:** question

---

### Summary

Three estimators score **higher on confident inputs**, while their docstrings
state that higher values indicate more uncertainty. The `Estimator` base class
documents the contract as "Higher values should indicate more uncertain samples",
and `PredictionRejectionArea` relies on it.

Either the sign or the documentation is wrong; which one is the intended
convention is a question for you rather than something to assume.

### Evidence 1 — synthetic distributions, no data or model

```python
import numpy as np
from lm_polygraph.estimators import RenyiNeg, FisherRao, SelfCertainty

V = 2000
def lp(kind):
    l = np.full(V, -20.0); l[7] = 0.0          # peaked / confident
    if kind == "uniform": l = np.zeros(V)
    l = l - np.log(np.exp(l).sum())
    return np.tile(l, (5, 1))

for est in (RenyiNeg(), FisherRao(), SelfCertainty()):
    c = est({"greedy_log_probs": [lp("confident")]})[0]
    u = est({"greedy_log_probs": [lp("uniform")]})[0]
    print(f"{est!s:16s} confident={c:9.4f}  uniform={u:9.4f}  "
          f"{'OK' if u > c else 'INVERTED'}")
```

```
RenyiNeg          confident= -12.8581  uniform= -15.2018  INVERTED
FisherRao         confident=   0.7995  uniform=   0.0000  INVERTED
SelfCertainty     confident= -12.3891  uniform=  -0.0000  OK
```

Both compute a divergence **from** the uniform distribution, which is maximal
when the model is confident. `FisherRao` is `2/pi * arccos(BC(p, uniform))`: `0`
for uniform, `~1` for a delta.

Docstrings: `renyi_neg.py:39` and `fisher_rao.py:36`, both "Higher values
indicate more uncertain samples."

### Evidence 2 — real data, n = 1000

Qwen2.5-3B-Instruct on TriviaQA, normalized PRR@0.5 as shipped versus negated:

| Estimator | as shipped | negated | rho vs `SelfCertainty` |
|---|---:|---:|---:|
| `FisherRao` | −1.0359 | **+0.8539** | −0.983 |
| `RenyiNeg` | −1.0419 | **+0.8313** | −0.983 |
| `MeanConditionalPointwiseMutualInformation` | −0.7616 | **+0.6401** | −0.621 |

A normalized PRR near `−1` is the signature of a correctly-ranking but inverted
score: it is as far below random as the oracle is above it. The first two also
correlate at `rho = -0.983` with `SelfCertainty`, which is correctly oriented.

`MeanConditionalPointwiseMutualInformation` correlates less strongly (`-0.621`)
but shows the same reversal under negation, so we flag it with weaker evidence.

### Question

Is the intended convention that these are *confidence* scores (in which case the
docstrings are wrong), or *uncertainty* scores (in which case the sign is)? We
have reported them as-shipped rather than silently flipping them, since
misrepresenting the library's estimators would be worse than reporting a negative
number.

Happy to open a PR either way once you say which.

### Environment

Evidence 1 needs none of this — it is CPU-only and runs anywhere. The following is
the environment in which the **Evidence 2** numbers were measured: a free Google
Colab T4 instance.

* lm-polygraph `efea882d810d07770e71d3a80e02416d09751435`
* transformers 4.50.0, torch 2.6.0+cu124
* Qwen/Qwen2.5-3B-Instruct, fp16, Tesla T4 (Google Colab), TriviaQA n=1000
