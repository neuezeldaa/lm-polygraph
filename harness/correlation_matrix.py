#!/usr/bin/env python3
"""Rank-correlation matrix over every estimator in a run.

PRR asks "which estimator ranks best". This asks the prior question: **how many
distinct measurements are on the table at all?** If the top three entries
correlate at rho > 0.9 they are three names for one signal, and their PRR
ordering is a coin flip rather than a result.

Spearman rho and Kendall tau are computed on the per-sample uncertainty vectors,
so this is invariant to any monotone rescaling -- exactly the equivalence class
PRR itself cares about.

Outputs a full CSV matrix plus a markdown digest: correlation clusters, the
cross-family pairs, and the near-duplicates of each top-ranked estimator.

No GPU: reads per_sample_<seed>.npz.

Usage:
    python harness/correlation_matrix.py --npz <per_sample_seed1.npz> \
        --out-md harness/correlations.md --out-csv harness/correlations.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines import prr_normalized  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--quality", default=None)
    ap.add_argument("--cluster-threshold", type=float, default=0.90,
                    help="|rho| at or above which two estimators are treated as "
                         "measuring the same thing.")
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--top", type=int, default=8,
                    help="How many top-PRR estimators to detail.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    ue = {k[4:]: np.asarray(d[k], dtype=np.float64) for k in d.files if k.startswith("ue::")}
    gm = {k[4:]: np.asarray(d[k], dtype=np.float64) for k in d.files if k.startswith("gm::")}
    qname = args.quality or next((k for k in gm if "Accuracy" in k), sorted(gm)[0])
    target = gm[qname]

    names = sorted(ue)
    ok = [n for n in names if np.isfinite(ue[n]).sum() > 10]
    m = len(ok)

    rho = np.eye(m)
    tau = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            a, b = ue[ok[i]], ue[ok[j]]
            msk = np.isfinite(a) & np.isfinite(b)
            rho[i, j] = rho[j, i] = spearmanr(a[msk], b[msk]).correlation
            tau[i, j] = tau[j, i] = kendalltau(a[msk], b[msk]).correlation

    prr = {}
    for n in ok:
        msk = np.isfinite(ue[n]) & np.isfinite(target)
        prr[n] = prr_normalized(ue[n][msk], target[msk])
    ranked = sorted(ok, key=lambda n: prr[n], reverse=True)

    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8") as fh:
            fh.write("estimator," + ",".join(ok) + "\n")
            for i, n in enumerate(ok):
                fh.write(n + "," + ",".join(f"{v:.4f}" for v in rho[i]) + "\n")
        print(f"[corr] wrote {args.out_csv}  ({m}x{m} Spearman)")

    L = []
    add = lambda s="": (print(s), L.append(s))
    idx = {n: i for i, n in enumerate(ok)}

    add(f"# Rank-correlation matrix ({m} estimators, n={target.size}, quality: {qname})")
    add()
    add("Spearman rho on the per-sample uncertainty vectors. PRR asks which estimator")
    add("ranks best; this asks how many *distinct* measurements exist. Two estimators")
    add(f"correlating at |rho| >= {args.cluster_threshold} are one signal under two names, and")
    add("their PRR ordering is not a result.")
    add()

    add(f"## Near-duplicates of the top {args.top} by normalized PRR@0.5")
    add()
    add("| Rank | Estimator | PRR@0.5 | correlates >= {:.2f} with |".format(args.cluster_threshold))
    add("|---:|---|---:|---|")
    for r, n in enumerate(ranked[: args.top], 1):
        dup = [f"{o} ({rho[idx[n], idx[o]]:.3f})"
               for o in ok if o != n and abs(rho[idx[n], idx[o]]) >= args.cluster_threshold]
        dup.sort(key=lambda s: -float(s.split("(")[1].rstrip(")")))
        add(f"| {r} | `{n}` | {prr[n]:.4f} | {', '.join(f'`{x}`' for x in dup) or '—'} |")
    add()

    # connected components at the threshold
    seen, clusters = set(), []
    for n in ranked:
        if n in seen:
            continue
        stack, comp = [n], set()
        while stack:
            c = stack.pop()
            if c in comp:
                continue
            comp.add(c)
            for o in ok:
                if o not in comp and abs(rho[idx[c], idx[o]]) >= args.cluster_threshold:
                    stack.append(o)
        seen |= comp
        clusters.append(sorted(comp, key=lambda x: -prr[x]))

    add(f"## Clusters at |rho| >= {args.cluster_threshold}")
    add()
    add(f"{len(clusters)} distinct signals among {m} estimators.")
    add()
    add("| # | Size | Best PRR | Members (by PRR) |")
    add("|---:|---:|---:|---|")
    for i, c in enumerate(clusters, 1):
        add(f"| {i} | {len(c)} | {prr[c[0]]:.4f} | " +
            ", ".join(f"`{x}`" for x in c) + " |")
    add()

    add("## Selected cross-family pairs")
    add()
    add("| A | B | rho | tau |")
    add("|---|---|---:|---:|")
    interesting = []
    for fam_a, fam_b in [("marginal", "spilled"), ("marginal", "logit"),
                         ("logit", "spilled"), ("marginal", "Pooled"),
                         ("marginal", "SelfCertainty"), ("marginal", "MeanTokenEntropy"),
                         ("marginal", "MaximumSequenceProbability")]:
        for a in ok:
            if fam_a not in a:
                continue
            for b in ok:
                if fam_b not in b or a == b:
                    continue
                if a.endswith("_noterm") != b.endswith("_noterm"):
                    continue
                interesting.append((a, b, rho[idx[a], idx[b]], tau[idx[a], idx[b]]))
    seen_pairs = set()
    for a, b, r, t in sorted(interesting, key=lambda x: -abs(x[2]))[:200]:
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if len(seen_pairs) > 18:
            break
        add(f"| `{a}` | `{b}` | {r:.3f} | {t:.3f} |")
    add()

    add("## Reading")
    add()
    top = ranked[0]
    dups = [o for o in ok if o != top and abs(rho[idx[top], idx[o]]) >= args.cluster_threshold]
    if dups:
        add(f"The top-ranked `{top}` is not a distinct measurement: it correlates at")
        add(f"|rho| >= {args.cluster_threshold} with {len(dups)} other estimator(s). Its margin over them is")
        add("not evidence that it measures something they do not.")
    else:
        add(f"The top-ranked `{top}` has no near-duplicate at this threshold.")

    if args.out_md:
        args.out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n[corr] wrote {args.out_md}")


if __name__ == "__main__":
    main()
