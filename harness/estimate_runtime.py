#!/usr/bin/env python3
"""Project a long run's wall clock from a short measured one.

Answers "am I looking at 30 minutes or three hours?" BEFORE the long run starts,
rather than at minute 80. Reads run_meta.json (written by run_baselines.py) from
one or more completed short runs and extrapolates linearly in n.

Linear in n is the right model here: cost is dominated by per-sample generation
and per-sample stat calculators, and every calculator in these configs is
per-batch work with no cross-sample term. The measured point already includes
fixed startup (model load, dataset download), so that fixed cost is
double-counted at larger n -- the projection is therefore a slight OVERestimate,
which is the safe direction.

Usage:
    python harness/estimate_runtime.py --from <dir> [--from <dir2>] --to-n 1000
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="sources", type=Path, action="append", required=True,
                    help="Run directory containing run_meta.json. Repeatable.")
    ap.add_argument("--to-n", type=int, default=1000)
    ap.add_argument("--label", action="append", default=None)
    args = ap.parse_args()

    labels = args.label or [s.name for s in args.sources]
    total_min = 0.0
    rows = []

    for src, label in zip(args.sources, labels):
        meta_path = src / "run_meta.json"
        if not meta_path.exists():
            print(f"[eta] WARNING: no run_meta.json in {src}; skipping")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n = meta.get("n")
        elapsed = meta.get("elapsed_sec")
        if not n or not elapsed:
            print(f"[eta] WARNING: {meta_path} lacks n/elapsed_sec; skipping")
            continue
        per_sample = elapsed / n
        projected = per_sample * args.to_n
        rows.append((label, n, elapsed, per_sample, projected))
        total_min += projected / 60

    if not rows:
        sys.exit("[eta] nothing to project from")

    print("=" * 74)
    print(f"RUNTIME PROJECTION  ->  n = {args.to_n}")
    print("=" * 74)
    print(f"{'run':<22}{'measured n':>11}{'measured':>11}{'sec/sample':>13}{'projected':>15}")
    for label, n, elapsed, per_sample, projected in rows:
        print(f"{label:<22}{n:>11}{elapsed/60:>9.1f}m{per_sample:>13.2f}{projected/60:>13.1f}m")

    print("-" * 74)
    print(f"{'TOTAL for all runs':<22}{'':>11}{'':>11}{'':>13}{total_min:>13.1f}m")
    print("=" * 74)

    if total_min > 150:
        print("\n  WARNING: over 2.5 hours total. A free Colab session will very likely")
        print("  be reclaimed first. Reduce n, or split the runs across sessions --")
        print("  each run writes its own manager to Drive, so they can be done apart.")
    elif total_min > 75:
        print("\n  NOTE: over an hour total. Keep the browser tab alive; each run's")
        print("  manager lands on Drive as it completes, so a loss costs one run, not all.")
    else:
        print("\n  Comfortably inside a free Colab session.")

    print("\n  (Linear in n; includes fixed startup at the measured point, so this")
    print("   slightly OVERestimates -- the safe direction.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
