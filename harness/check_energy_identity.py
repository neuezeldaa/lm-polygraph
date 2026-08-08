#!/usr/bin/env python3
"""Permanent gate: does log p = E^m - E^l hold on real run data?

Algebraically exact. But the two sides come from DIFFERENT forward passes --
greedy_log_likelihoods from incremental decoding with a KV cache, the energies
from a full-sequence teacher-forced prefill -- so in fp16 they diverge. The
residual therefore measures how far apart those two numerical paths are, which is
exactly the quantity dE inherits and amplifies.

Why it matters: dE = Z_{j+1} - theta_j is a cancelling difference. Measured on
Qwen2.5-3B/TriviaQA, |theta| ~ 22.7 and |Z| ~ 27.1 give |dE| ~ 4.2, an
amplification of ~6.5x. A 0.19 nat logit-path divergence becomes ~1.2 nats in dE
before max pooling picks the worst token -- which is why runs that differ only in
batch size or attention kernel disagreed on dE at rho=0.29.

Reads per_sample_<seed>.npz; no GPU. Reconstructs mean(theta) and mean(Z) from
the mean-pooled logit and marginal variants, so it needs those in the run.

Usage:
    python harness/check_energy_identity.py --npz <per_sample_seed1.npz> [--max-residual 0.05]
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--max-residual", type=float, default=0.05,
                    help="Fail above this mean |residual| in nats.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    ue = {k[4:]: np.asarray(d[k], dtype=np.float64) for k in d.files if k.startswith("ue::")}

    need = ["SpilledEnergy_logit_mean", "SpilledEnergy_marginal_mean", "Perplexity"]
    missing = [n for n in need if n not in ue]
    if missing:
        print(f"[identity] SKIP: run lacks {missing}")
        return 0

    theta = -ue["SpilledEnergy_logit_mean"]     # mean theta[id]
    Z = -ue["SpilledEnergy_marginal_mean"]      # mean Z
    logp = -ue["Perplexity"]                    # mean log p, from generation
    resid = np.abs((theta - Z) - logp)

    dE = np.abs(ue.get("SpilledEnergy_spilled_mean", np.array([np.nan])))
    amp = np.median(np.abs(Z)) / np.median(dE) if np.isfinite(dE).any() else np.nan

    print("=" * 68)
    print("ENERGY IDENTITY GATE   log p == E^m - E^l")
    print("=" * 68)
    print(f"  median |theta[id]|      : {np.median(theta):8.3f}")
    print(f"  median |Z|              : {np.median(Z):8.3f}")
    print(f"  median |dE| (mean pool) : {np.median(dE):8.3f}")
    print(f"  amplification |Z|/|dE|  : {amp:8.1f}x")
    print(f"  mean |residual|         : {resid.mean():8.4f} nats")
    print(f"  max  |residual|         : {resid.max():8.4f} nats")
    print(f"  implied dE error        : ~{resid.mean()*amp:.2f} nats")
    print("-" * 68)

    if resid.mean() > args.max_residual:
        print(f"  FAIL: mean residual {resid.mean():.4f} > {args.max_residual}")
        print("  The teacher-forced pass and the generation disagree by more than")
        print("  fp16 noise should allow. Either the projection is not being done in")
        print("  float32 (EnergyCalculator(fp32_projection=True)), or the energies")
        print("  are misaligned. dE inherits this error amplified, so do NOT report")
        print("  any dE-based result from this run.")
        print("=" * 68)
        return 1

    print("  PASS")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
