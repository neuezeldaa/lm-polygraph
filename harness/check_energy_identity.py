#!/usr/bin/env python3
"""Two gates on the energy statistics: one exact, one calibrated.

HARD GATE (exact by construction, no tolerance to choose)
    Within a single teacher-forced pass, ``tok - lse`` is ``log_softmax(theta)``
    at the sampled token, so it is a log-probability and must be ``<= 0`` for
    every token, with ``exp(tok - lse) <= 1``. A violation means the statistics
    are misaligned or corrupted, not merely imprecise. Uses the per-token
    ``energy_token_logits`` / ``energy_lse`` persisted via ``save_stats``.

CALIBRATED GATE (threshold measured, never guessed)
    The cross-pass residual ``|(tok - lse) - greedy_log_likelihoods|`` compares a
    teacher-forced prefill against incremental decoding with a KV cache. Those are
    different numerical paths, so in fp16 a nonzero residual is CORRECT behaviour,
    not a defect. Picking a threshold by intuition risks the same false-positive
    gate we already had to remove once from the A-vs-C comparison.

    So the floor is measured from a reference run in the cleanest configuration
    available -- batch_size=1 (no padding) with fp32_projection on -- and the
    threshold is set above that floor with an explicit margin. The derivation is
    printed so the number in the report can be traced.

Why any of this matters: ``dE = Z_{j+1} - theta_j`` is a cancelling difference
(|theta| ~ 22.7, |Z| ~ 27.1, |dE| ~ 4.2 on Qwen2.5-3B/TriviaQA), so it inherits
~6.5x of any logit error, and ``max`` pooling then selects the noisiest token.

Usage:
    # calibrate from the bs=1 run, then gate the others
    python harness/check_energy_identity.py --run <A dir> --floor-run <C dir>
    # or gate against an explicit threshold
    python harness/check_energy_identity.py --run <A dir> --max-residual 0.08
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

MARGIN = 3.0        # threshold = floor * MARGIN
MIN_THRESHOLD = 0.02  # never gate tighter than this, whatever the floor says


def _field(man, name):
    return man.get(name) if isinstance(man, dict) else getattr(man, name, None)


def load_run(run_dir: Path):
    import torch

    mans = sorted(glob.glob(str(run_dir / "ue_manager_seed*")))
    if not mans:
        sys.exit(f"[identity] no ue_manager_seed* in {run_dir}")
    blob = torch.load(mans[0], weights_only=False)
    stats = _field(blob, "stats") or {}
    return {
        "dir": run_dir,
        "tok": stats.get("energy_token_logits"),
        "lse": stats.get("energy_lse"),
        "ll": stats.get("greedy_log_likelihoods"),
    }


def within_pass_violations(run):
    """Exact invariant: tok - lse must be <= 0 everywhere."""
    tok, lse = run["tok"], run["lse"]
    if not tok or not lse:
        return None
    worst, n_bad, n_tot = 0.0, 0, 0
    for t, l in zip(tok, lse):
        t = np.asarray(t, dtype=np.float64)
        l = np.asarray(l, dtype=np.float64)[: len(t)]
        if t.size == 0:
            continue
        logp = t - l
        n_tot += logp.size
        n_bad += int((logp > 1e-4).sum())
        worst = max(worst, float(logp.max()))
    return {"n_bad": n_bad, "n_total": n_tot, "worst_logp": worst}


def cross_pass_residual(run):
    """|(tok - lse) - greedy_log_likelihoods|, per token, across the run."""
    tok, lse, ll = run["tok"], run["lse"], run["ll"]
    if not tok or not lse or not ll:
        return None
    res = []
    for t, l, g in zip(tok, lse, ll):
        t = np.asarray(t, dtype=np.float64)
        l = np.asarray(l, dtype=np.float64)[: len(t)]
        g = np.asarray(g, dtype=np.float64)
        k = min(len(t), len(g))
        if k:
            res.append(np.abs((t[:k] - l[:k]) - g[:k]))
    if not res:
        return None
    a = np.concatenate(res)
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="Run directory to gate.")
    ap.add_argument("--floor-run", type=Path, default=None,
                    help="Reference run for calibration; use the batch_size=1 run.")
    ap.add_argument("--max-residual", type=float, default=None,
                    help="Explicit threshold; overrides calibration.")
    ap.add_argument("--margin", type=float, default=MARGIN)
    args = ap.parse_args()

    run = load_run(args.run)
    print("=" * 70)
    print(f"ENERGY GATES  --  {args.run}")
    print("=" * 70)

    failures = []

    # ---- hard gate ---------------------------------------------------------
    wp = within_pass_violations(run)
    print("\n[hard] within-pass invariant   tok - lse <= 0   (exact by construction)")
    if wp is None:
        print("  SKIP: energy_token_logits / energy_lse not in save_stats")
    else:
        print(f"  tokens checked : {wp['n_total']}")
        print(f"  violations     : {wp['n_bad']}")
        print(f"  worst log p    : {wp['worst_logp']:+.6f}  (must be <= 0)")
        if wp["n_bad"]:
            failures.append(
                f"{wp['n_bad']}/{wp['n_total']} tokens have tok - lse > 0, i.e. a "
                f"log-probability above 1 (worst {wp['worst_logp']:+.4f}). The "
                "energies are misaligned or corrupted, not merely imprecise."
            )
        else:
            print("  PASS")

    # ---- calibrated gate ---------------------------------------------------
    cp = cross_pass_residual(run)
    print("\n[calibrated] cross-pass residual   |(tok - lse) - greedy_log_likelihoods|")
    print("  compares a teacher-forced prefill against incremental decoding, so a")
    print("  nonzero residual is correct behaviour in fp16, not a defect.")
    if cp is None:
        print("  SKIP: needs energy stats and greedy_log_likelihoods")
    else:
        print(f"  mean {cp['mean']:.4f}   p95 {cp['p95']:.4f}   max {cp['max']:.4f}"
              f"   (n={cp['n']} tokens)")

        threshold, how = args.max_residual, "given explicitly"
        if threshold is None and args.floor_run is not None:
            floor = cross_pass_residual(load_run(args.floor_run))
            if floor is None:
                print(f"  WARNING: no energy stats in {args.floor_run}; cannot calibrate")
            else:
                threshold = max(floor["mean"] * args.margin, MIN_THRESHOLD)
                how = (f"calibrated: floor {floor['mean']:.4f} (mean residual of "
                       f"{args.floor_run.name}, the batch_size=1 / fp32_projection "
                       f"reference) x margin {args.margin} , min {MIN_THRESHOLD}")
        if threshold is None:
            print("  no threshold: pass --floor-run or --max-residual to gate")
        else:
            print(f"  threshold      : {threshold:.4f}")
            print(f"  derivation     : {how}")
            if cp["mean"] > threshold:
                failures.append(
                    f"cross-pass residual {cp['mean']:.4f} > {threshold:.4f}. dE "
                    f"inherits ~6.5x of this, so dE-based results are not reportable."
                )
            else:
                print("  PASS")

    print("\n" + "=" * 70)
    if failures:
        print("VERDICT: FAIL")
        for f in failures:
            print(f"  - {f}")
        print("=" * 70)
        return 1

    # A gate that skipped has not passed. Reporting PASS when nothing was
    # actually checked is the silent-skip failure mode this whole harness exists
    # to avoid.
    if wp is None and cp is None:
        print("VERDICT: INCONCLUSIVE -- nothing was checked")
        print("  The run has no energy_token_logits / energy_lse in its stats, so")
        print("  neither gate could run. Add them to save_stats and re-run; do not")
        print("  read this as a pass.")
        print("=" * 70)
        return 2

    checked = [n for n, g in (("within-pass", wp), ("cross-pass", cp)) if g is not None]
    print(f"VERDICT: PASS ({', '.join(checked)} checked)")
    if len(checked) < 2:
        print("  NOTE: the other gate skipped; this is a partial pass.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
