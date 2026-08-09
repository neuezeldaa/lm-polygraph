# Spilled Energy in lm-polygraph — implementation and evaluation

**Method:** Spilled Energy (Minut, Dewidar & Masi, ICLR 2026)
**Author:** Roman Zolotov
**Deliverables:** PR `spilled-energy` (implementation + tests), `spilled-energy-experiments` (harness, configs, notebook), this report.

> Every number in this report is reproducible from the artifacts in `runs/`. Where a result depends on the run configuration, both configurations are given. Normalized PRR values may differ from a fresh recomputation in the fourth decimal, since the normalization draws a random reference; rankings and confidence intervals are unaffected.

---

## 1. Summary

Spilled Energy reinterprets an LLM's final softmax as an energy-based model and derives two training-free quantities from the raw logits — a *logit energy* `E^l` and a *marginal energy* `E^m` — whose difference across adjacent decoding steps, the *spilled energy* `ΔE`, should vanish in theory but does not in practice. The paper reports that this residual correlates with factual errors.

I implemented the method end-to-end in lm-polygraph and evaluated it on TriviaQA against a baseline set derived mechanically from the library's own shipped estimator list. Three findings:

1. **The method's strongest components do not outperform the baselines they are compared against.** Best energy variant `SpilledEnergy_logit_max_noterm` reaches normalized PRR@0.5 of **0.845 [0.779, 0.903]**; `MeanTokenEntropy` reaches **0.900 [0.847, 0.946]**. The confidence intervals overlap almost entirely.
2. **`E^l` and `E^m` are not distinguishable from each other**, correlating at Spearman ρ = 0.987–0.999 under matched pooling, and `E^m` is not distinguishable from `SelfCertainty` (ρ = 0.985). The two energies whose difference defines the method measure, on this data, the same thing.
3. **`ΔE` is numerically ill-conditioned in fp16 by construction, and it underperforms the energies it is built from.** It subtracts two quantities of magnitude ≈ 25 to obtain one of magnitude ≈ 4. Reproducibility across run configurations degrades monotonically with how much cancellation and selection a quantity involves, and at the extreme (`ΔE` under `max` pooling) two runs over *identical generations* disagree by more than the spread of the quantity itself. The underperformance survives in the cleanest configuration available, so it is a property of the method and not only of the arithmetic.

A secondary result concerns methodology rather than the method: **the definition of the pooling window changes which estimator wins.** Including two service tokens (newline + EOS) in the window moves pooled mean log-likelihood from −0.100 to **0.803** normalized PRR. Any comparison that does not fix this is not measuring what it claims to.

---

## 2. Experimental setup

| | |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct`, fp16 |
| Hardware | NVIDIA T4 (16 GB), free Colab |
| Dataset | TriviaQA, 5-shot continuation prompt |
| Decoding | greedy, `max_new_tokens=20`, `stop_strings=["\n"]` |
| n | 1000 (dev pass at 150) |
| Quality metric | exact-match `Accuracy` = **0.2930** |
| UQ metric | normalized PRR at rejection rate 0.5, 1000 bootstrap resamples |
| Attention impl. | `sdpa` for the main runs; `eager` at batch size 1 for the attention-based baselines (see §5.4) |

**Model choice.** The original plan named Llama-3.2-3B-Instruct. It was replaced with Qwen2.5-3B-Instruct because the Llama repository is gated behind an access approval whose latency is outside my control, and because a non-gated model lets a reviewer reproduce the notebook with no credentials at all. The two are comparable in size and fp16 footprint. This weakens direct comparability with the paper's own table, which uses LLaMA, Mistral and Gemma; §4 treats that explicitly.

**Prompt-format validation.** The paper localizes its `[u, w]` span by "prompting the LLM for a brief answer". The configuration above is the structural equivalent, and I verified rather than assumed it: 150/150 generations are single-line, median length 4 tokens, 0.7 % hit the token ceiling, 0 % contain special tokens, and the automatic verdict is that `answer window == generation` is defensible for this dataset. The check re-runs on every pass (`harness/validate_answer_span.py`).

---

## 3. Q1 — What is the core idea?

Write the next-token conditional as a ratio of two Boltzmann terms. From the softmax,

```
-log p(x_i | x_{i-1:1})  =  -θ(x_{i-1:1})[id(x_i)]  +  log Σ_k exp θ(x_{i-1:1})[k]
                            └────── E^l ──────┘      └──────── -E^m ────────┘
```

so `E^l` is the negated logit of the sampled token and `E^m` is the negated log-partition over the vocabulary. Both come from the same logit vector, but they are read differently: `E^l` at a single index, `E^m` by marginalizing over the whole vocabulary.

Applying the chain rule over a sequence, the total energy telescopes and two terms from adjacent steps meet. These should cancel — they are the same quantity viewed from two neighbouring positions — and in a trained LLM they do not, because they are produced by different components at different decoding steps. That residual is the **spilled energy**, and the paper's claim is that its magnitude tracks factual error.

**An index-convention discrepancy, worth stating because it inverts the sign.** Eq. (8) as printed pairs `E^l` at step `i+1` with `E^m` at step `i`. The authors' released code computes the opposite pairing, `delta = lse[j+1] − logit[j]`, which in the notation above is `E^l` at step `j` against `E^m` at step `j+1`:

```
paper, Eq. (8):   ΔE = E^l(x_{i+1:1}) − E^m(x_{i:1})
authors' code:    ΔE = E^l(x_{i:1})   − E^m(x_{i+1:1})     ← implemented here
```

The two differ by a sign, and an inverted uncertainty score produces strongly negative PRR rather than a merely weak one — a failure mode that is easy to mistake for a broken implementation. I followed the code, and pinned the convention with a hand-computed unit test rather than deriving it from the paper.

Three quantities are proposed as detectors: marginal energy `E^m`, spilled energy `ΔE`, and scaled spilled energy `ΔE_s = |E^m| · ΔE`. `E^l` alone is the classical "logit confidence" baseline. All are training-free and derived from the logits of a single forward pass — though recovering them inside lm-polygraph costs a second pass, for the reason below.

**Why this needs a new StatCalculator rather than an estimator alone.** `WhiteboxModel._ScoresProcessor` (`utils/model.py:455`) applies `log_softmax`, and `generate()` writes its output over `out.scores` (`utils/model.py:535`); `GreedyProbsCalculator` then stores what it is handed. The consequence is that `logsumexp(row) == 0` for every stored row by construction — the partition constant is destroyed before any calculator sees it, and cannot be recovered. `E^m` is exactly that constant. The implementation therefore contributes `EnergyCalculator`, which performs a teacher-forced pass and reduces raw logits on the fly to `energy_token_logits` (length N) and `energy_lse` (length N+1); the extra entry is what makes the adjacent-step difference defined at the last token. A unit test asserts `|logsumexp| > 0`, which fails loudly if the calculator is ever re-wired to normalized log-probabilities.

---

## 4. Q2 — Can the paper's results be reproduced?

**Not directly, and the mismatch is qualitative rather than a matter of degree.**

Three differences make this a partial reproduction by construction, all forced by the assignment or the hardware:

- **Metric.** The paper reports AUROC. The assignment requires normalized PRR at 0.5. These rank estimators differently in general.
- **Model.** The paper uses LLaMA-3-8B-Instruct, LLaMA, Mistral-Instruct and Qwen-3-8B. A T4 at fp16 admits ~3B parameters.
- **Precision.** §6.1 shows this is not a detail for this particular method.

With those caveats, the *ordering within the method's own variants* should still transfer, and it does not. On TriviaQA with LLaMA-Instruct the paper reports:

| Variant | Pool | AUROC |
|---|---|---:|
| Spilled `ΔE` | **Min** | **87.07** |
| Marginal `E^m` | Max | 80.13 |
| Marginal `E^m` | Min | 73.38 |
| Logit `E^l` | Max | 68.89 |
| Spilled `ΔE_s` | Max | 48.70 |
| p(true) | — | 45.99 |

In my runs the ordering is close to inverted. The `marginal` and `logit` families occupy the top of the table (0.59–0.85 normalized PRR) while every `spilled` variant sits well below them (−0.13 to 0.53), and `spilled_min` at 0.1764 — the paper's best configuration on this exact dataset — is among the weakest of the 35 estimators evaluated. `ΔE_s`, which the paper ranks last on TriviaQA, likewise ranks near the bottom here; that single agreement is the only part of the ordering that transfers.

**One structural observation about the paper's comparison.** Its baselines are `p(true)` (average AUROC 51.29 for LLaMA-Instruct — barely above chance) and the trained probing classifiers of Orgad et al. It does not compare against the standard training-free UQ baselines that lm-polygraph ships: token entropy, self-certainty, sequence probability. §5 shows that this is where the method's advantage disappears. Reproducing the paper's *numbers* and reproducing its *conclusion* are separable questions, and the second is the one that fails here.

---

## 5. Q3 — How does normalized PRR compare with the baselines?

### 5.1 Baseline selection is rule-based, not hand-picked

To avoid any suspicion of a favourable baseline set, the candidates are the 50 rows of upstream's `examples/configs/estimators/default_estimators.yaml`, and exclusions are made by mechanical predicates over each estimator's *resolved transitive dependency set* — no name lists. `harness/derive_tiers.py` emits the full table (`harness/estimator_tiers.csv`).

| Tier | Count | Excluded because |
|---|---:|---|
| `needs_sampling` | 29 | Requires sampled generations; different compute class |
| `single_pass_cheap` | 10 | **kept** |
| `needs_train_data` | 5 | Fits statistics on a background split — the assignment forbids supervised training |
| `requires_bs1_under_fp16` | 4 | **kept**, run separately (§5.4) |
| `needs_external_corpus` | 1 | `Focus` downloads a 401 MB IDF corpus |
| `single_pass_plus_aux_model` | 1 | **kept**, flagged (CCP; needs DeBERTa-MNLI) |

Forward-pass counts are derived per row by source inspection across the MRO, so the "matched compute budget" claim is a column in the table rather than an assertion: `PTrue`, `MeanPMI`, `MeanConditionalPMI` and `AttentionScore` cost two passes; everything else costs one.

### 5.2 Main result

Normalized PRR@0.5, n = 1000, 95 % bootstrap CI:

| Estimator | nPRR@0.5 | 95 % CI |
|---|---:|---|
| `MeanTokenEntropy` | 0.8997 | [0.847, 0.946] |
| `SpilledEnergy_logit_max` (no terminator) | 0.8452 | [0.779, 0.903] |
| `SpilledEnergy_marginal_mean` | 0.8410 | [0.780, 0.896] |
| `SelfCertainty` | 0.8398 | [0.776, 0.894] |
| `SpilledEnergy_marginal_max` | 0.8229 | [0.749, 0.883] |
| `SpilledEnergy_marginal_mean` (no terminator) | 0.7672 | [0.692, 0.834] |
| `MaximumSequenceProbability` | 0.5452 | [0.440, 0.644] |
| `SpilledEnergy_spilled_max` | 0.5290 | [0.430, 0.623] |
| `PTrue` | 0.4514 | [0.348, 0.558] |
| `SpilledEnergy_spilled_mean` | 0.3387 | [0.225, 0.452] |
| `SpilledEnergy_spilled_min` | 0.1763 | [0.053, 0.293] |
| `CCP` | −0.0508 | [−0.170, 0.066] |
| `Perplexity` | −0.1003 | [−0.218, 0.024] |

**Does the method outperform the baselines? No.** The best energy variant is statistically indistinguishable from the best baseline; the method's own headline quantity `ΔE` is separated from the top of the table by non-overlapping intervals in the wrong direction. The whole table was produced twice, in independent runs over identical generations, and reproduces to the fourth decimal.

### 5.3 The ablation ladder, and why the comparison needed one

`ΔE` benefits from two things at once — the energy formulation *and* localization to the answer span with pooling. A win would be unattributable. Run B therefore computes every rung under the same window and the same three poolings, so adjacent rungs differ by exactly one ingredient:

```
pooled log-likelihood  →  E^l  →  E^m  →  ΔE  →  ΔE_s
```

Pooled log-likelihood and pooled entropy required a new `PooledBaseline` control, since no estimator in lm-polygraph exposes a pooling parameter. Two identities validate it on real data rather than on a stub: `Pooled_entropy_mean` = `MeanTokenEntropy` = 0.8997 and `Pooled_log_likelihood_mean` = `Perplexity` = −0.1003, exactly. The control reproduces the library's own estimators when configured to match, so it is not subtly different from the thing it controls for.

**The rank-correlation matrix reframes the question.** Among 35 estimators there are only **14 distinct signals** at |ρ| ≥ 0.9, and the entire top of the table is a single ten-member cluster containing `MeanTokenEntropy`, `SelfCertainty`, every `marginal_*` and the `logit_*` no-terminator variants. Within it:

| A | B | ρ |
|---|---|---:|
| `marginal_min_noterm` | `logit_min_noterm` | 0.999 |
| `marginal_mean_noterm` | `logit_mean_noterm` | 0.998 |
| `marginal_max_noterm` | `logit_max_noterm` | 0.987 |
| `marginal_mean` | `SelfCertainty` | 0.985 |
| `marginal_mean` | `MeanTokenEntropy` | 0.902 |

The first three rows are the load-bearing ones. `E^l` and `E^m` are the two quantities whose difference the entire derivation rests on, and under matched pooling they are one measurement — at ρ = 0.999 the difference between them is essentially all that survives, and §6.1 argues that what survives is largely numerical. `ΔE` *is* a distinct signal (ρ = 0.112 against `marginal_mean`, 0.608 in the no-terminator window) — it is simply a worse one.

### 5.4 Attention-based baselines

`RAUQ`, `AttentionScore` and `CSL` require attention weights, which forces `output_attentions=True`, which in turn disables a guard in `transformers` that is needed under fp16 (§8.3). They were therefore run at batch size 1 with the `eager` implementation, at n = 300:

| Estimator | nPRR@0.5 | 95 % CI |
|---|---:|---|
| `RAUQ (entropy)` | 0.9152 | [0.830, 0.976] |
| `RAUQ` | 0.9038 | [0.815, 0.969] |
| `SpilledEnergy_logit_mean` | 0.8938 | [0.806, 0.968] |
| `SpilledEnergy_marginal_mean` | 0.8829 | [0.791, 0.964] |
| `CSL` | 0.8303 | [0.711, 0.932] |
| `SpilledEnergy_spilled_mean` | 0.4838 | [0.273, 0.683] |
| `AttentionScore (layer=18)` | 0.3456 | [0.146, 0.538] |
| `SpilledEnergy_spilled_max` | 0.1707 | [−0.044, 0.383] |

Four energy variants are carried in this run purely as anchors, so that the two configurations can be compared on the same estimator rather than across disjoint sets. §6.1 uses them.

The headline here is that two attention-based estimators — one forward pass, no sampling, no second model — sit at the top of everything measured in this project. They are excluded from the main table only because of the fp16 interaction in §8.3, not on merit.

### 5.5 Run comparability

PRR is only comparable across tables if the generations are identical. Runs A and B are byte-identical by SHA-256 over `greedy_texts` and `greedy_tokens`. Run C shares generations with A on 298/300 samples (0.67 %), within the 2 % tolerance set for it; the two differences are expected `sdpa`-versus-`eager` rounding on near-tied argmax, which is why that gate is a tolerance and not an equality.

---

## 6. Q4 — Limitations and failure modes

### 6.1 `ΔE` is ill-conditioned in fp16 — the dominant failure mode

Measured on run A: median `|θ[id]|` = 22.72, median `|Z|` = 27.09, median `|ΔE|` = 4.19. The definition subtracts two quantities of magnitude ≈ 25 to produce one of magnitude ≈ 4, so **any logit error is amplified ≈ 6.5×**, and `max` pooling then selects the noisiest token in the window.

The consequence is measurable as a dose–response relationship. Comparing the same estimators over the same 300 samples with *identical generations*, in two runs differing only in batch size and attention kernel:

| Quantity | cancellation | selection | ρ(bs=4, bs=1) | std of disagreement |
|---|---|---|---:|---:|
| `marginal_mean` | none | none | 0.956 | 0.95 |
| `logit_mean` | none | none | 0.820 | 2.11 |
| `spilled_mean` | yes | none | 0.736 | 1.16 |
| `spilled_max` | yes | yes | **0.293** | **3.32** |

Agreement degrades monotonically with how much subtraction and how much extremum selection a quantity involves. For `spilled_max` the spread of the disagreement (3.32) exceeds the spread of the quantity itself (3.17 and 2.67 in the two runs), and there is a systematic +3.06 offset with the batched run higher in 81.3 % of samples — consistent with `max` pooling under noise, where the noisier pass selects a higher maximum. At that point the estimator partly reports *how noisy the forward pass was* rather than how uncertain the model was.

The same effect is visible directly in the energy statistics. The cross-pass residual of the identity `log p = E^m − E^l` is **0.0368** at batch size 1 and **0.1993** at batch size 4 — **5.4× worse under batching**, on identical generations. The within-pass identity `tok − lse` passes at both batch sizes, so the energies are structurally correct: what differs is precision, not correctness.

**Does the weakness survive better conditioning?** Recomputing both configurations on the same 300 samples with one formula:

| Quantity | bs = 4 | bs = 1 | Δ |
|---|---:|---:|---:|
| `marginal_mean` | 0.775 | 0.730 | −0.045 |
| `logit_mean` | 0.551 | 0.730 | +0.180 |
| `spilled_mean` | 0.191 | 0.506 | **+0.315** |
| `spilled_max` | 0.258 | 0.214 | −0.045 |

`ΔE` is substantially *understated* by the batched run — `spilled_mean` nearly triples at batch size 1 — but even in the cleanest configuration available it reaches 0.506 against 0.730 for the two energies it is derived from. **The underperformance is real; the batched measurement additionally penalises it.** Both halves matter: the first is the finding, the second is a warning that any single reported number for `ΔE` is configuration-dependent to a degree the paper does not discuss.

**A mitigation that did not work, recorded as such.** `fp32_projection` recomputes the vocabulary projection in float32 for the few rows needed (~13 MB), on the hypothesis that fp16 accumulation in the `lm_head` was the dominant error term. It is not: with it enabled the energies moved by 0.002 on average and the residual was unchanged (0.18947 vs 0.1895 pooled). The divergence originates in the fp16 hidden states of the transformer body, which the projection cannot reach and which cannot be raised to fp32 for a 3B model on a 16 GB T4. The option remains enabled because it is strictly more accurate at negligible cost, but it is reported as a measured negative result.

This is a property of the method under the assignment's fp16 constraint, not of this implementation.

### 6.2 The window definition dominates the comparison

The pooling window is not neutral. With a median generation length of 4 tokens, the trailing newline and EOS occupy **half** the window. Measured at n = 1000:

| Estimator | incl. terminator | excl. terminator | Δ |
|---|---:|---:|---:|
| `Pooled_log_likelihood_mean` | −0.1003 | 0.8034 | **+0.904** |
| `SpilledEnergy_logit_max` | 0.4659 | 0.8452 | +0.379 |
| `SpilledEnergy_marginal_mean` | 0.8410 | 0.7672 | −0.074 |
| `SpilledEnergy_spilled_max` | 0.5290 | 0.1143 | −0.415 |

Two service tokens move a baseline by almost a full unit of normalized PRR, and they move it in the *opposite* direction to the energy variants. I adopted the excluded-terminator window as primary because the terminator is not part of the answer and the paper's construct is defined on the answer span — noting that this choice costs the method 0.074 and rescues the baseline by 0.904, i.e. it is the window that disfavours my own implementation.

### 6.3 Redundancy with existing estimators

`E^m` correlates with `SelfCertainty` at ρ = 0.985. Whatever `E^m` captures, lm-polygraph already shipped it under another name. The energy derivation is a different *route* to the quantity, not a different quantity.

### 6.4 Answer-span localization is load-bearing and fragile

The paper's `[u, w]` span is obtained by prompting for a brief answer. On TriviaQA that is defensible and I validated it numerically (§2). It will not transfer to datasets where the answer is embedded in longer output, and locating the span by matching against the gold answer — the obvious alternative — is label leakage, since ground truth is unavailable at inference. This bounds where the method can be applied without a separate span detector.

---

## 7. Q5 — How could the method be improved?

**7.1 Report a conditioning diagnostic alongside `ΔE`.** The cancellation ratio `|ΔE| / max(|E^l|, |E^m|)` is computable per sample at no cost. Where it is small, `ΔE` is a difference of nearly equal numbers and its value should be treated as unreliable rather than as low uncertainty. Used as a gate, this converts the dominant failure mode into an explicit abstention. The identity residual (§6.1) serves the same purpose at the level of a whole run, and is what revealed the 5.4× batching penalty.

**7.2 Compute the energies in higher precision, and say which part.** `E^m` is a log-partition over a 151 936-token vocabulary. Projecting it in fp32 costs ~13 MB and is strictly more accurate, but measurement showed the `lm_head` accumulation is *not* the dominant error term (§6.1) — the divergence originates in the fp16 hidden states of the body. A method whose central quantity is a near-cancellation should state its precision requirements rather than leave them to the implementer, and should say which stage they apply to.

**7.3 Compare against the right baselines.** The paper's comparison set is `p(true)` and trained probes. Against token entropy and self-certainty the advantage does not survive. A revision that includes these — and reports the rank correlation with them — would make a much stronger claim if it holds, and would be more useful if it doesn't.

**7.4 Separate `E^l` from `E^m` before differencing them.** At ρ ≥ 0.987 under matched pooling the two energies are one measurement, so their difference is dominated by whatever makes them differ — which on this evidence is largely numerical. Identifying a regime where they genuinely decouple (longer answers, higher-entropy tasks, larger models) would be the most direct way to rescue the construct.

**7.5 Fix the window definition explicitly in the protocol.** Given §6.2, any published comparison should state which tokens enter the pooling window. This is cheap, and without it the reported ordering is not reproducible even in principle.

---

## 8. Findings about lm-polygraph

Four issues surfaced during this work. They are reported as observations, not as patches — the PR touches no upstream behaviour beyond four single-line registrations.

**8.1 Non-finite logits are silently replaced by a uniform distribution.** `_SanitizeLogitsProcessor` rewrites a fully non-finite row to zeros. Downstream, `greedy_log_likelihoods` are then finite and constant at exactly `−ln V = −11.9312`, and the generation decodes as token id 0 repeated. A numerically unstable run therefore produces plausible-looking numbers instead of an error, and the failure is invisible in precisely the quantities a user would check. This is the mechanism behind the first failed run in this project (72 % of samples corrupted, accuracy exactly 0.000).

**8.2 Three estimators are oriented opposite to their documentation.** `RenyiNeg`, `FisherRao` and `MeanConditionalPointwiseMutualInformation` score *higher* on confident inputs, while their docstrings state that higher values indicate more uncertainty. Evidence differs by estimator: the first two correlate with `SelfCertainty` at ρ = −0.983 and −0.983, and on synthetic distributions with no data at all `FisherRao` returns 0.7995 for a confident distribution and 0.0000 for a uniform one. `MeanConditionalPointwiseMutualInformation` correlates at only −0.621, but its normalized PRR moves from −0.7616 to +0.6401 under negation, which is the same diagnosis by a different route. All three are reported as-shipped in §5.2; negated values are in the appendix. Either the sign or the documentation is wrong; which one is a question for the maintainers.

**8.3 fp16 + `eager` + `output_attentions=True` + left padding produces NaN.** Qwen2 attention computes `attn_weights + causal_mask` with `causal_mask = finfo(dtype).min`; in fp16, `−30 + (−65504)` overflows to `−inf` and softmax yields NaN. `transformers` guards this via `AttentionMaskConverter._unmask_unattended`, but the guard is gated on `sdpa` *and* `not output_attentions` — both violated by any attention-based UQ estimator. This is why §5.4 exists.

**8.4 `PromptCalculator` is registered twice** at `register_default_stat_calculators.py:102` and `:107`. Benign: the dictionary is keyed by stat name, so the second registration overwrites the first idempotently, and the calculator is constructed and run exactly once — verified empirically, not only by reading. It is dead code rather than duplicated computation. Confirmed pre-existing via `git stash`.

---

## 9. Reproducing this

**The deliverable is split across two branches, deliberately.** `spilled-energy`
is the PR: the `EnergyCalculator`, the `SpilledEnergy` estimator, their tests, and
the configs needed to run them — self-contained, and the only upstream edits are
four single-line registrations. `spilled-energy-experiments` adds the harness, the
notebook, and this report.

The split is not cosmetic. The ablation ladder (run B) and the terminator A/B load
estimators by the dotted path `harness.pooled_baseline`, and `harness/` has no
place in a library PR — shipping those configs in the PR would give a maintainer an
`ImportError` on first use. They therefore live on the experiments branch together
with the module they need. **Run B and the terminator A/B are reproduced from the
experiments branch, not from the PR**; runs A and C reproduce from either.

```bash
git clone --branch spilled-energy-experiments https://github.com/neuezeldaa/lm-polygraph
# open notebooks/spilled_energy_colab_t4.ipynb in Colab, select a T4, Run all
```

The notebook is generated by `harness/make_notebook.py` and needs no manual editing. It gates on, in order: model provenance (hard assert on the resolved model path), degeneracy (no generation may be a single repeated token, nothing pinned at `−ln V`), exact-match accuracy in [0.1, 0.9], answer-span validation, cross-run comparability, and the energy identity against a floor calibrated at batch size 1.

Gates are invoked through `harness/gate.py`, which runs each command as a subprocess and raises on a non-zero exit. This is deliberate and was not the original design: the first version called the gates with the notebook shell escape, which does not halt on failure, so every gate was advisory. One gate was additionally invoked with a stale argument name and had been failing silently. Both are fixed, and the halting behaviour is verified against real data — the energy identity gate passes on the batch-size-1 run and stops the notebook on the batched ones.

Unit tests added by this work — CPU only, ~15–30 s, no GPU and no model download:

```bash
pytest test/test_spilled_energy.py test/test_generation_trimming.py \
       test/test_batch_invariance.py test/test_energy_batch_invariance.py
```

(The full `pytest test/` also collects upstream tests, which download `bloomz-560m`.)

---

## Appendix pointers

- `runs/A_baselines_n1000/prr_0.5_table.md` — full 35-row main table
- `runs/B_ladder_n1000/prr_0.5_table.md` — full 36-row ablation ladder
- `runs/*/correlations.md` — rank-correlation matrix and cluster decomposition
- `runs/AB_terminator_n150/terminator_ab.md` — paired terminator comparison
- `harness/estimator_tiers.csv` — all 50 upstream estimators with resolved dependencies, tier and exclusion reason
