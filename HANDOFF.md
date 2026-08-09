# HANDOFF — Spilled Energy on lm-polygraph

For a fresh session. Facts are recoverable from the code and `runs/`; **decisions
and dead ends are not**, and re-litigating them is the main way to waste time
here. This document is those.

**The empirical work is closed. No re-runs are needed or wanted.**

---

## 1. State

| | |
|---|---|
| Base | `upstream/main` = `efea882d810d07770e71d3a80e02416d09751435` |
| PR branch | `spilled-energy` = `d39a04ff` — 21 files, **+1998, 0 deletions** |
| Experiments branch | `spilled-energy-experiments` = `44dd0d2e` |
| Remotes | `origin` = `neuezeldaa/lm-polygraph` (fork, public) · `upstream` = `IINemo/lm-polygraph`, **push disabled** |

`origin` originally pointed at *upstream* — the mapping was inverted and a naive
`git push origin` would have targeted the maintainers' repo. Fixed; upstream's
push URL is set to an invalid string on purpose.

**Final runs** (all in `runs/`, all with `fp32_projection`, all gated):

| Run | config | n | bs | attn | role |
|---|---|---:|---:|---|---|
| A | `eval_triviaqa_qwen` | 1000 | 4 | sdpa | primary table |
| B | `eval_triviaqa_ladder` | 1000 | 4 | sdpa | ablation ladder |
| C | `eval_triviaqa_attention_bs1` | 300 | 1 | eager | attention baselines + 4 anchors; **calibration floor** |
| AB | `eval_triviaqa_terminator_ab` | 150 | 4 | sdpa | terminator A/B |

`REPORT.md` (experiments branch) answers the five assignment questions with every
number verified against `runs/`. Two known wrong cross-references, both
two-character fixes not yet applied: line 80 cites §5.1 where it means **§6.1**;
line 162 cites §7.4 where it means **§8.3**.

---

## 2. Decisions, with the reasons that are not in the code

**Qwen2.5-3B-Instruct, not Llama-3.2-3B.** Llama is gated behind an approval
whose latency is outside our control, and a week-long deadline should not sit
behind someone else's moderation queue. The fork is public and the notebook needs
no HF credentials, so a reviewer reproduces it with nothing. Cost: weaker
comparability with the paper's own table. Do not "fix" this by switching back.

**Excluded-terminator window is primary.** Chosen *against* the method's
interest: excluding costs `marginal_mean` 0.8678 → 0.7575 and rescues
`Pooled_log_likelihood_mean` from 0.0541 → 0.7931. It is nonetheless the faithful
window — the terminator is not part of the answer and the paper's construct is
defined on the answer span. The included variants are retained as a labelled
secondary set so the delta stays visible. **This reasoning is the point**; a
session that re-derives only the numbers will be tempted to pick the flattering
window.

**ΔE follows the authors' released code, not the printed Eq. (8).** They
disagree. Code: `delta = lse[j+1] - logit[j]` = `E^l` at the earlier step minus
`E^m` at the later. The paper's equation pairs them the other way, which inverts
the sign and yields a strongly *negative* PRR that reads as a broken
implementation. `REPORT.md` §3 shows both forms rather than silently resolving
them; a hand-computed unit test pins the convention.

**Baselines derived mechanically.** `harness/derive_tiers.py` classifies all 50
rows of upstream's `default_estimators.yaml` by predicates over each estimator's
*resolved transitive dependency set* — never a name list — so the comparison set
cannot be called cherry-picked. Tiers: 29 sampling, 10 single-pass-cheap, 5
train-data, 4 bs1-under-fp16, 1 external-corpus, 1 plus-aux-model.

**A/B/C split and the two gate strengths.** A and B share an attention
implementation and differ only in their estimator list, so identical generations
are a real requirement → **hard SHA-256 gate**. C uses `eager` (forced: sdpa
cannot return attention weights), and different kernels round differently in
fp16, so near-tied argmax can legitimately flip → **2 % tolerance**. A byte
gate there would fail on correct behaviour. Observed: 2/300 = 0.67 %.

---

## 3. Traps already paid for

**`_SanitizeLogitsProcessor` masks NaN as a uniform distribution.** A fully
non-finite logit row becomes all zeros, so `log_softmax` gives `-ln(V) =
-11.9312` at every position and argmax is token 0 (`!` in Qwen). Every recorded
diagnostic looks finite and healthy. **A finiteness check on `greedy_log_probs`
can never detect this** — the sanitizer runs before those values are stored. We
wrongly concluded "no overflow" from finite log-likelihoods; recognising
`-11.9312` as `-ln(V)` is what cracked it.

**Its trigger:** fp16 + `eager` + left padding. Qwen2 computes
`attn_weights + finfo(dtype).min`; in fp16 `-30 + (-65504)` overflows to `-inf`
and softmax gives NaN. `transformers` guards this via `_unmask_unattended` but
gates it on `sdpa` **and** `not output_attentions` — both violated by any
attention-based estimator. Hence C runs at bs=1 (no padding → no fully-masked
row). Cost the first run: 108/150 corrupted, accuracy exactly 0.000.

**`eos_token_id` vs `pad_token_id`.** Qwen2.5-Instruct declares `eos_token_id`
= `<|im_end|>` while continuation-style completions end with `<|endoftext|>`,
which is *also* `pad_token_id`. `GreedyProbsCalculator` trims by scanning for the
single declared id, so nothing was trimmed: every sequence looked pinned to the
20-token ceiling, `<|endoftext|>` survived into `greedy_texts` making exact match
impossible, and padding entered the energy window. Fixed in our own loader
(upstream uses the same extension point for Gemma). **`stop_strings` was working
all along** — that was a misdiagnosis.

**Gates were advisory, not enforcing.** Colab's `!command` never raises, so
`Run all` sailed past every failed gate and printed tables anyway. One gate was
additionally called with a stale flag and had been failing silently. Fixed:
`harness/gate.py` runs each command as a subprocess and raises. Do not reintroduce
`!{cmd}` for anything that gates.

**`fp32_projection` is a measured negative result, not a fix.** Introduced on the
hypothesis that fp16 `lm_head` accumulation dominated the error. It does not:
energies moved ~0.002 mean and the residual was unchanged (0.18947 vs 0.1895).
The divergence is in the fp16 hidden states of the body, unreachable without
running the body in higher precision, which does not fit a T4. Left enabled
because it is strictly more accurate for ~13 MB. **What actually moves the
residual is batch size**: 0.0368 at bs=1 vs 0.1993 at bs=4.

**The left-padding hypothesis was investigated and refuted.** `EnergyCalculator`
pads on the **right**, so HF's default `position_ids` are already correct and
causality keeps pads out of the slice. `test_energy_batch_invariance` passes in
float32 *and* float16. The A↔C disagreement is conditioning, not misalignment:
`|θ|≈22.7`, `|Z|≈27.1`, `|ΔE|≈4.2` → ~6.5× amplification, and `max` pooling
selects the noisiest token. Do not re-open this as a padding bug.

---

## 4. Open items

1. **REPORT.md is committed** (`44dd0d2e`). Only the two cross-references above
   remain.
2. **Three upstream issue drafts, split, awaiting review before filing:**
   `harness/UPSTREAM_ISSUE_1_sanitizer.md` (lm-polygraph's bug — the one worth
   filing), `..._2_attention_fp16.md` (framed as *documentation*: the defect is
   transformers' gating, lm-polygraph only exposes it), `..._3_orientation.md`
   (`RenyiNeg`/`FisherRao`/`MeanCondPMI` inverted; framed as a *question*, since
   the intended convention is theirs to state).
3. **The PR is deliberately unopened** and stays that way until Roman says
   otherwise.

### A note on what the PR branch may contain

`ablation_ladder.yaml` and `terminator_ab.yaml` were removed from the PR branch:
they load estimators by the dotted path `harness.pooled_baseline`, and `harness/`
exists only on the experiments branch, so a maintainer running them from the PR
would hit an `ImportError`. Anything added to `configs/` on the PR branch must be
self-contained -- check with `grep -rn "name: harness\." configs/` before
committing there.

---

## 5. Standing rules

* Commit **and push** before reporting anything done. Colab pulls from GitHub;
  local-only work cannot be run.
* Never force-push or rewrite history on either branch.
* Never run `gh pr create` or any equivalent.
* If a push fails, say so loudly — do not report success.
* Colab's "Save a copy in GitHub" has twice pushed the notebook back and blocked
  a push. The notebook is **generated** by `harness/make_notebook.py`; resolve any
  conflict by regenerating, never by hand-editing the JSON.
* Upstream code is not modified beyond four single-line registrations. Flag
  upstream bugs; do not patch them.
