# CLAUDE.md — Spilled Energy on lm-polygraph (working conventions)

Assignment: implement *Spilled Energy* (ICLR 2026) as an lm-polygraph estimator
and compare against baselines on **normalized PRR@0.5**.

**Read `HANDOFF.md` first** (repository root). It carries the decisions, the
reasons behind them, and the traps already paid for. This file is conventions
only, and it is the authoritative copy — `../CLAUDE.md` is only a pointer here.

**The empirical work is closed.** Runs A, B, C and the terminator A/B are final;
no re-runs are needed or wanted.

## Branches
- **`spilled-energy`** — PR candidate. StatCalculator, Estimator, tests, configs.
  Nothing from `harness/` or `notebooks/`. Currently 21 files, +1819 / −0 over
  `upstream/main` (`efea882d`).
- **`spilled-energy-experiments`** — the above **plus** `harness/`, `notebooks/`,
  `REPORT.md`, `HANDOFF.md`. This is what Colab clones.
- Remotes: `origin` = the fork; `upstream` = `IINemo/lm-polygraph` with push
  disabled. The mapping was originally inverted — do not "restore" it.

## Hard constraints
- Training-free, logits-only. No probes, ensembles, or sampling for the method.
- Free Colab/Kaggle **T4, 16 GB, fp16** (Turing: no hardware bf16).
- `sdpa` + `output_attentions: false` for the main runs — **both** are required
  for correctness, not performance (see HANDOFF §3). `sdpa` is PyTorch-native,
  not FlashAttention-2, so it is inside the constraint.
- **Single reported metric: normalized PRR@0.5** → `prr_0.5_normalized`.
- Do **not** modify upstream estimators, calculators or metric code. Flag bugs;
  don't patch. The only upstream edits are four single-line registrations.

## Frozen experiment settings
`Qwen/Qwen2.5-3B-Instruct` fp16 · TriviaQA continuation, test, 5-shot, multiref ·
greedy, `max_new_tokens=20`, stop on `"\n"` · `AccuracyMetric` (normalized exact
match) · seed 1 · terminator **excluded** from the pooling window (primary),
included retained as a labelled secondary set.

## Conventions
- One command reproduces a run: `python harness/run_baselines.py --config <cfg>`.
  `--skip-run` is a first-class CPU-only entry point for offline analysis.
- The notebook is **generated** by `harness/make_notebook.py` and validated with
  `nbformat`. Never hand-edit it; resolve conflicts by regenerating.
- Gates run through `harness/gate.py`, which raises on nonzero exit. Never use a
  bare `!{cmd}` for anything that gates — it does not halt the notebook.
- Exit-code convention: `0` pass, `1` fail, `2` inconclusive (nothing checked —
  **not** a pass).
- Cache generations and per-sample arrays to disk so analysis never re-runs the
  LLM.
- Verify claims against code and artifacts before asserting them. Several
  confident diagnoses in this project were wrong and were caught only by
  measurement.

## Tests
CPU-only, seconds, no GPU:
```bash
pytest test/test_spilled_energy.py test/test_generation_trimming.py \
       test/test_batch_invariance.py test/test_energy_batch_invariance.py
```
(The full `test/` also collects upstream tests that download `bloomz-560m`.)

## Standing rules
- Commit **and push** before reporting anything done.
- Never force-push or rewrite history on either branch.
- Never run `gh pr create`. The PR stays unopened until Roman says otherwise.
- If a push fails, say so loudly.
