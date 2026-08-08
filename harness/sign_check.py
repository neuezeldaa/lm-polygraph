#!/usr/bin/env python3
"""Recompute normalized PRR@0.5 for every estimator AND its negation.

A wrong sign convention does not look like noise -- it looks like a method that
works in reverse, giving a large NEGATIVE normalized PRR. This reads the
per-sample arrays a run already saved and reports both orientations side by side,
so an orientation error is caught before hours of GPU time rather than after.

No GPU, no model, no rerun: everything comes from per_sample_<seed>.npz.

Usage:
    python harness/sign_check.py --npz <per_sample_seed1.npz> [--quality Accuracy]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines import prr_normalized  # noqa: E402  (same PRR as the tables)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--quality", default=None,
                    help="Quality key (default: the one containing 'Accuracy').")
    ap.add_argument("--flag-threshold", type=float, default=0.05,
                    help="Report an estimator as INVERTED when negating it improves "
                         "normalized PRR by more than this.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    ue = {k[len("ue::"):]: d[k] for k in d.files if k.startswith("ue::")}
    gm = {k[len("gm::"):]: d[k] for k in d.files if k.startswith("gm::")}
    if not ue or not gm:
        sys.exit(f"{args.npz} has no ue::/gm:: arrays (found {d.files[:5]})")

    qname = args.quality or next(
        (k for k in gm if "Accuracy" in k), sorted(gm)[0]
    )
    target = np.asarray(gm[qname], dtype=np.float64)
    print(f"quality : {qname}   n={target.size}   mean={target.mean():.4f}")
    print(f"source  : {args.npz}\n")

    rows = []
    for name, vals in ue.items():
        v = np.asarray(vals, dtype=np.float64)
        mask = np.isfinite(v) & np.isfinite(target)
        if mask.sum() < 10:
            continue
        a = prr_normalized(v[mask], target[mask])
        b = prr_normalized(-v[mask], target[mask])
        rows.append((name, a, b, mask.sum()))

    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"{'estimator':<44}{'as-is':>10}{'negated':>10}{'better':>9}  flag")
    print("-" * 84)
    inverted = []
    for name, a, b, n in rows:
        better = "negated" if b > a else "as-is"
        flag = ""
        if b - a > args.flag_threshold:
            flag = "INVERTED"
            inverted.append((name, a, b))
        print(f"{name:<44}{a:>10.4f}{b:>10.4f}{better:>9}  {flag}")

    print("\n" + "=" * 84)
    if inverted:
        print(f"{len(inverted)} estimator(s) score materially better negated:")
        for name, a, b in sorted(inverted, key=lambda r: r[2] - r[1], reverse=True):
            print(f"  {name:<44} {a:>8.4f} -> {b:>8.4f}   (+{b-a:.4f})")
    else:
        print("No estimator improves materially when negated: every sign convention holds.")
    print("=" * 84)

    # A near-perfect mirror is the fingerprint of a PURE orientation error rather
    # than a weak-but-correctly-oriented score -- but only when the as-is value is
    # the negative one. A strong estimator is also near-symmetric under negation,
    # so requiring a < 0 keeps correctly oriented methods out of this list.
    mirrors = [
        (n, a, b) for n, a, b, _ in rows
        if a < 0 and abs(a + b) < 0.25 and abs(a) > 0.3
    ]
    if mirrors:
        print("\nNear-perfect mirrors with a NEGATIVE as-is score -- pure orientation errors")
        print("(the score ranks correctly, just upside down):")
        for n, a, b in mirrors:
            print(f"  {n:<44} {a:>8.4f} / {b:>8.4f}")


if __name__ == "__main__":
    main()
