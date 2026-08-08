#!/usr/bin/env python3
"""Pair up the terminator-included vs terminator-excluded results side by side.

The A/B run puts both settings in ONE UEManager, so they share generations
exactly and the only difference is the pooling window. This reads the per-sample
arrays and reports each estimator's normalized PRR@0.5 under both settings with
the delta, grouped by pooling so the min/max/mean pattern is visible.

No GPU: reads per_sample_<seed>.npz from the completed A/B run.

Usage:
    python harness/terminator_ab_report.py --npz <per_sample_seed1.npz>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines import prr_normalized, bootstrap_ci  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--quality", default=None)
    ap.add_argument("--n-boot", type=int, default=0,
                    help="Bootstrap resamples for CIs (0 = skip; slow at n=150).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    ue = {k[4:]: d[k] for k in d.files if k.startswith("ue::")}
    gm = {k[4:]: d[k] for k in d.files if k.startswith("gm::")}
    qname = args.quality or next((k for k in gm if "Accuracy" in k), sorted(gm)[0])
    target = np.asarray(gm[qname], dtype=np.float64)

    def score(name):
        v = np.asarray(ue[name], dtype=np.float64)
        m = np.isfinite(v) & np.isfinite(target)
        if m.sum() < 10:
            return None, None
        val = prr_normalized(v[m], target[m])
        ci = bootstrap_ci(v[m], target[m], n_boot=args.n_boot) if args.n_boot else None
        return val, ci

    pairs = []
    for name in ue:
        if name.endswith("_noterm"):
            continue
        if f"{name}_noterm" in ue:
            pairs.append(name)

    lines = []
    add = lambda s="": (print(s), lines.append(s))

    add(f"# Terminator A/B — normalized PRR@0.5   (quality: {qname}, n={target.size})")
    add()
    add("The pooling window either includes the trailing terminator (newline + EOS)")
    add("or excludes it. Both settings come from ONE run, so generations are identical")
    add("and the window is the only difference.")
    add()
    ci_h = "  | 95% CI incl | 95% CI excl" if args.n_boot else ""
    add(f"| Estimator | incl. terminator | excl. terminator | delta{ci_h} |")
    add("|---|---:|---:|---:|" + ("---:|---:|" if args.n_boot else ""))

    rows = []
    for name in pairs:
        a, cia = score(name)
        b, cib = score(f"{name}_noterm")
        if a is None or b is None:
            continue
        rows.append((name, a, b, b - a, cia, cib))

    for name, a, b, delta, cia, cib in sorted(rows, key=lambda r: r[2], reverse=True):
        extra = ""
        if args.n_boot:
            fmt = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if c and np.isfinite(c[0]) else "—"
            extra = f" | {fmt(cia)} | {fmt(cib)}"
        add(f"| {name} | {a:.4f} | {b:.4f} | {delta:+.4f}{extra} |")

    add()
    add("## By pooling")
    add()
    add("| Pooling | mean delta | max improvement | max regression |")
    add("|---|---:|---:|---:|")
    for pl in ("min", "max", "mean"):
        sub = [r for r in rows if r[0].endswith(f"_{pl}")]
        if not sub:
            continue
        deltas = [r[3] for r in sub]
        best = max(sub, key=lambda r: r[3])
        worst = min(sub, key=lambda r: r[3])
        add(f"| {pl} | {np.mean(deltas):+.4f} | {best[3]:+.4f} ({best[0]}) | "
            f"{worst[3]:+.4f} ({worst[0]}) |")

    add()
    if not rows:
        add("**No paired estimators found.** The run must contain each estimator")
        add("twice, identical except for `exclude_terminator`. Check that the config")
        add("is `estimators: terminator_ab` and that n is large enough to score.")
        if args.out:
            args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 1

    deltas = [r[3] for r in rows]
    big = [r for r in rows if abs(r[3]) >= 0.05]
    add(f"**Overall:** mean delta {np.mean(deltas):+.4f}, "
        f"largest |delta| {max(abs(x) for x in deltas):.4f}, "
        f"{len(big)}/{len(rows)} estimators move by >= 0.05.")
    add()
    if max(abs(x) for x in deltas) < 0.05:
        add("Effect is NEGLIGIBLE: no estimator moves by 0.05. Keeping the terminator")
        add("in the window costs nothing; one sentence in the report suffices.")
    else:
        add("Effect is MATERIAL: at least one estimator moves by >= 0.05, so the")
        add("window definition is doing real work and belongs in the ablation section")
        add("as a finding, not a footnote.")

    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[ab] wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
