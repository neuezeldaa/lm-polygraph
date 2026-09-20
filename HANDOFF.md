# HANDOFF — Spilled Energy on lm-polygraph

For a fresh session. Facts are recoverable from the code and `runs/`; **decisions
and dead ends are not**, and re-litigating them is the main way to waste time
here. This document is those.

**The empirical work was closed; the PR review reopened it.** ArtemVazh requested changes (remove the extra forward pass, fix the prompt formatting in instruct mode, re-run on the instruct dataset with AlignScore), and checking those turned up a padding artefact that invalidates part of §6.1 of the report. One more GPU pass is needed. The plan is in `C:\Users\Roman\.claude\plans\valiant-booping-chipmunk.md`.

---

## 1. State

| | |
|---|---|
| Base | `upstream/main` = `efea882d810d07770e71d3a80e02416d09751435` |
| PR branch | `spilled-energy` = `22b3469a` — squashed, 15 files, **+1435, 0 deletions**; pre-squash history kept at `backup/spilled-energy-presquash` = `d39a04ff` (21 files, +1819) |
| Experiments branch | `spilled-energy-experiments` — current tip; no hash recorded here, since it moves with every documentation edit including this one. Check with `git rev-parse spilled-energy-experiments` |
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
number verified against `runs/`. All cross-references resolve, and §9
opens by explaining the two-branch split.

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
The reason is now known — the residual it targeted was dominated by pad rows (see
below), not by precision at all. In the reworked implementation, which reads the
generation's own logits instead of re-running the model, the option disappears.

**The A↔C disagreement is a pad in the pooling window, not fp16 conditioning.**
This one cost the most and was wrong twice, so read it carefully before touching
§6.1 of the report:

* The *first* hypothesis, left-padding misalignment inside `EnergyCalculator`,
  was correctly refuted: it pads on the **right**, HF's `position_ids` are
  correct, and `test_energy_batch_invariance` passes in float32 and float16.
* The *second*, that the residual disagreement was fp16 ill-conditioning
  amplified by `ΔE`, was **wrong**. `GreedyProbsCalculator` trims at the first
  EOS inclusive; with `eos_token_id == pad_token_id` a sequence that finishes
  early at bs>1 keeps the first pad `generate` appended. Run A (bs=4) and run C
  (bs=1) share texts on 298/300 samples but token windows on only 100/300, and
  198/300 differ by exactly that pad, whose median `log p` is −21.95.
* Split by that grouping, the disagreement is entirely the pad: on identical
  windows ρ(A, C) = 1.000 for all four anchors, and the identity residual on
  real tokens is 0.0393 (bs=4) vs 0.0368 (bs=1) — 1.07×, not the 5.4× that the
  pooled figure suggested.
* The primary `_noterm` scores are unaffected (pad counts as a terminator and is
  dropped): recomputed on exactly the 198 affected samples, all twelve variants
  reproduce at ρ ≥ 0.9999.

The `|θ|≈22.7`, `|Z|≈27.1`, `|ΔE|≈4.2` amplification (~6.5×) is still real
arithmetic — it just is not what was being measured. Reproduce the pad on CPU
with the tiny stub by giving it `stop_strings` that fire at different steps;
without a stop condition no sequence finishes early, which is why
`test_batched_generation_matches_individual` never caught it.

---

## 4. Open items

1. **REPORT.md has been corrected for the pad artefact** (§1 item 3, §5.3, §5.5,
   §6.1 rewritten, §6.2 caveat, §7.1, §7.2, new §8.4). The numbers in §5.2 and §5.3
   stand — they are measured in the pad-free window and that was verified, not
   assumed. What changed is the *attribution* in §6.1. Once the instruct re-run
   lands, the tables need refreshing with its numbers.
2. **Three upstream issue drafts, split, awaiting review before filing:**
   `harness/UPSTREAM_ISSUE_1_sanitizer.md` (lm-polygraph's bug — the one worth
   filing), `..._2_attention_fp16.md` (framed as *documentation*: the defect is
   transformers' gating, lm-polygraph only exposes it), `..._3_orientation.md`
   (`RenyiNeg`/`FisherRao`/`MeanCondPMI` inverted; framed as a *question*, since
   the intended convention is theirs to state).
3. **The PR is open and has a review from ArtemVazh (changes requested).** Three
   requests: (a) drop the extra forward pass — have `GreedyProbsCalculator` ask for
   `output_logits=True` and take the raw logits from the generation pass;
   (b) `EnergyCalculator._prompt_ids` bypasses `WhiteboxModel.tokenize()` and so
   skips the chat template when `instruct=True` (measured: 11 tokens vs 40 for the
   same TriviaQA prompt); (c) re-run with `instruct: true`, the `simple_instruct`
   dataset and AlignScore, based on
   `examples/configs/polygraph_eval_triviaqa_simple_instruct.yaml`. All three are
   correct. **Do not post to the PR or push to the PR branch without Roman's
   explicit yes** — he posts the replies himself; the draft is prepared for him.
4. **When the PR description is written, it must state the branch split and the
   reason.** The PR is self-contained for the method and its tests; the ablation
   ladder, the terminator A/B and `harness/` live on
   `spilled-energy-experiments` because they load estimators by the dotted path
   `harness.pooled_baseline`, which has no place in a library PR. Said plainly,
   deliberate separation reads very differently from an oversight — and a
   reviewer who notices run B is absent will otherwise assume the latter.
   REPORT.md §9 carries the same note for readers of the report alone.

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
* Upstream code is not modified beyond four single-line registrations — **with
  one exception the reviewer asked for**: `GreedyProbsCalculator` now also
  requests `output_logits` and emits the two reduced energy statistics. Anything
  beyond that (notably the trim in §3) stays a flagged observation until a
  maintainer asks for a patch.
