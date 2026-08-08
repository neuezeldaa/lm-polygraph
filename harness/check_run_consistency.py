#!/usr/bin/env python3
"""Check that two runs saw the same generations, and fail loudly when they did not.

Tables from separate UEManager runs are only comparable if both saw the same
generations -- otherwise the quality vector differs, the PRR normalisation
differs, and the two tables silently stop being about the same thing.

Greedy decoding with a fixed seed *should* guarantee that, but it is verified
rather than assumed: runs can resolve different stat calculators, and fp16
reductions are not associative.

Two modes, because not every pair of runs CAN be byte-identical.

STRICT (default, --max-mismatch-frac 0) -- for runs sharing an attention
implementation, e.g. the primary baseline run vs the ablation ladder. They differ
only in their estimator list, so identical generations are a real requirement and
any difference is a bug. Compares sha256 of greedy_texts and greedy_tokens, and
the quality vector elementwise.

SOFT (--max-mismatch-frac > 0) -- for runs that legitimately differ, e.g. the
primary run (sdpa) vs the attention run (eager, forced because sdpa cannot return
attention weights). Different kernels round differently in fp16, so where the top
two candidates are nearly tied the argmax can flip. Byte equality would fail on
correct behaviour. This mode reports the mismatch count and fraction, shows
examples, and fails only above the given tolerance: a handful is kernel noise,
several percent means something is actually wrong.

Exit code 0 = comparable. Exit code 1 = DO NOT compare these tables.

Usage:
    # strict, for A vs B (same attention implementation)
    python harness/check_run_consistency.py --a <dirA> --b <dirB>

    # soft, for A vs C (sdpa vs eager)
    python harness/check_run_consistency.py --a <dirA> --b <dirC>
        --allow-prefix --max-mismatch-frac 0.02
"""

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _field(man, name):
    if isinstance(man, dict):
        return man.get(name)
    return getattr(man, name, None)


def load(save_dir: Path):
    import torch

    mans = sorted(glob.glob(str(save_dir / "ue_manager_seed*")))
    if not mans:
        sys.exit(f"[consistency] FATAL: no ue_manager_seed* in {save_dir}")
    blob = torch.load(mans[0], weights_only=False)
    stats = _field(blob, "stats") or {}
    gen_metrics = _field(blob, "gen_metrics") or {}

    quality, qname = None, None
    for k, v in gen_metrics.items():
        name = k[1] if isinstance(k, tuple) and len(k) == 2 else str(k)
        if "Accuracy" in name:
            quality, qname = np.asarray(v, dtype=np.float64), name
            break
    if quality is None and gen_metrics:  # fall back to the first metric present
        k, v = next(iter(gen_metrics.items()))
        qname = k[1] if isinstance(k, tuple) and len(k) == 2 else str(k)
        quality = np.asarray(v, dtype=np.float64)

    return {
        "path": mans[0],
        "texts": stats.get("greedy_texts"),
        "tokens": stats.get("greedy_tokens"),
        "quality": quality,
        "quality_name": qname,
    }


def sha(obj) -> str:
    if obj is None:
        return "MISSING"
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="First run directory.")
    ap.add_argument("--b", type=Path, required=True, help="Second run directory.")
    ap.add_argument("--label-a", default="run A")
    ap.add_argument("--label-b", default="run B")
    ap.add_argument("--max-mismatch-frac", type=float, default=0.0,
                    help="0 (default) = strict byte equality, for runs sharing an "
                         "attention implementation. A positive value switches to a "
                         "SOFT comparison, for runs that legitimately differ: sdpa "
                         "and eager are different kernels and round differently in "
                         "fp16, so a near-tied argmax can flip on a few samples.")
    ap.add_argument("--allow-prefix", action="store_true",
                    help="Runs may differ in n; compare their shared prefix. Valid "
                         "because Dataset.subsample is prefix-stable under a fixed seed.")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)

    print("=" * 70)
    print("RUN CONSISTENCY CHECK")
    print("=" * 70)
    print(f"  {args.label_a}: {A['path']}")
    print(f"  {args.label_b}: {B['path']}")

    failures = []

    na = len(A["texts"]) if A["texts"] else 0
    nb = len(B["texts"]) if B["texts"] else 0
    print(f"\n  n samples          : {na}  vs  {nb}")
    if na == 0 or nb == 0:
        failures.append("a run has no generations")

    # Runs of different n are still comparable on their overlap: Dataset.subsample
    # takes np.random.choice(N, size) under a fixed seed, and numpy's without-
    # replacement choice is prefix-stable, so the smaller run's samples are
    # exactly the first k of the larger run's. Verified in
    # test_subsample_is_prefix_stable. This is what lets the batch_size=1
    # attention run be checked against the batch_size=4 primary run.
    k = min(na, nb)
    if na != nb:
        if not args.allow_prefix:
            failures.append(
                f"sample count differs ({na} vs {nb}); pass --allow-prefix to "
                "compare the shared prefix instead"
            )
        else:
            print(f"  comparing the shared prefix of {k} samples "
                  f"(runs differ in n by design)")
    for d in (A, B):
        for key in ("texts", "tokens", "quality"):
            if d[key] is not None:
                d[key] = d[key][:k]

    ha, hb = sha(A["texts"]), sha(B["texts"])
    ta, tb = sha(A["tokens"]), sha(B["tokens"])
    print(f"  sha256 greedy_texts : {ha[:16]}...  vs  {hb[:16]}...")
    print(f"  sha256 greedy_tokens: {ta[:16]}...  vs  {tb[:16]}...")

    mismatched = [
        i for i, (x, y) in enumerate(zip(A["texts"] or [], B["texts"] or [])) if x != y
    ]
    frac = (len(mismatched) / k) if k else 1.0
    print(f"  generations differing: {len(mismatched)}/{k} = {frac:.2%}")

    strict = args.max_mismatch_frac <= 0
    if strict:
        if ha != hb:
            failures.append("greedy_texts differ")
        if ta != tb:
            failures.append("greedy_tokens differ")
    elif frac > args.max_mismatch_frac:
        failures.append(
            f"{len(mismatched)}/{k} generations differ ({frac:.2%}), above the "
            f"tolerance of {args.max_mismatch_frac:.2%}"
        )

    qa, qb = A["quality"], B["quality"]
    if qa is None or qb is None:
        failures.append("a quality vector is missing")
    else:
        print(f"\n  quality metric      : {A['quality_name']}  vs  {B['quality_name']}")
        print(f"  quality mean        : {qa.mean():.6f}  vs  {qb.mean():.6f}"
              f"   (delta {abs(qa.mean()-qb.mean()):.6f})")
        if qa.shape != qb.shape:
            failures.append(f"quality shape differs ({qa.shape} vs {qb.shape})")
        elif not np.array_equal(qa, qb):
            n_diff = int((qa != qb).sum())
            msg = (f"quality vector differs in {n_diff}/{qa.size} positions "
                   f"(means {qa.mean():.6f} vs {qb.mean():.6f})")
            if strict:
                failures.append(msg)
            else:
                print(f"  NOTE: {msg}")

    # ascii() so a narrow console codepage cannot raise here and swallow the verdict
    if mismatched:
        print(f"\n  first differing generations (of {len(mismatched)}):")
        for i in mismatched[:5]:
            print(f"    [{i}] {args.label_a}: {ascii(A['texts'][i])}")
            print(f"    [{i}] {args.label_b}: {ascii(B['texts'][i])}")

    print("\n" + "=" * 70)
    if failures:
        print("VERDICT: NOT COMPARABLE")
        for f in failures:
            print(f"  - {f}")
        if strict:
            print(
                "\nThese two runs share an attention implementation, so their\n"
                "generations must match exactly. A difference means the runs resolved\n"
                "different stat calculators or different generation settings. Fix the\n"
                "configs and re-run; do not place the tables side by side."
            )
        else:
            print(
                "\nToo many generations differ to attribute this to kernel rounding.\n"
                "Check that the two configs agree on model, decoding parameters, seed\n"
                "and subsample, and that the degeneracy gate passed for BOTH runs --\n"
                "a collapsed run will differ from a healthy one almost everywhere."
            )
        print("=" * 70)
        return 1

    if strict:
        print("VERDICT: COMPARABLE -- generations and quality are identical.")
    else:
        print(f"VERDICT: COMPARABLE -- {len(mismatched)}/{k} generations differ "
              f"({frac:.2%}), within the {args.max_mismatch_frac:.2%} tolerance.")
        print("  Expected: these runs use different attention kernels (sdpa vs eager),")
        print("  which round differently in fp16, so a near-tied argmax can flip.")
        print("  The tables remain comparable, but report this fraction alongside them.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
