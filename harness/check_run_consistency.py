#!/usr/bin/env python3
"""Assert that two runs produced BYTE-IDENTICAL generations, and fail loudly if not.

The baseline table and the ablation-ladder table come from two separate
UEManager runs. They are only comparable if both saw exactly the same
generations -- otherwise the quality vector differs, the PRR normalisation
differs, and the two tables silently stop being about the same thing.

Greedy decoding with a fixed seed *should* guarantee this. But the two runs do
not resolve identical stat calculators (the baseline run captures attention and
loads an NLI model; the ladder run does not), which can change kernel selection,
and fp16 reductions are not associative. So this is verified, not assumed.

Checks, all hard failures:
  * same number of samples
  * sha256 of greedy_texts identical
  * sha256 of greedy_tokens identical
  * quality (Accuracy) vector identical elementwise, and its mean identical

Exit code 0 = comparable. Exit code 1 = DO NOT compare these tables.

Usage:
    python harness/check_run_consistency.py --a <dir> --b <dir>
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
    if na != nb or na == 0:
        failures.append(f"sample count differs ({na} vs {nb})")

    ha, hb = sha(A["texts"]), sha(B["texts"])
    print(f"  sha256 greedy_texts: {ha[:16]}...  vs  {hb[:16]}...")
    if ha != hb:
        failures.append("greedy_texts differ")

    ta, tb = sha(A["tokens"]), sha(B["tokens"])
    print(f"  sha256 greedy_tokens: {ta[:16]}...  vs  {tb[:16]}...")
    if ta != tb:
        failures.append("greedy_tokens differ")

    qa, qb = A["quality"], B["quality"]
    if qa is None or qb is None:
        failures.append("a quality vector is missing")
    else:
        print(f"\n  quality metric     : {A['quality_name']}  vs  {B['quality_name']}")
        print(f"  quality mean       : {qa.mean():.6f}  vs  {qb.mean():.6f}")
        if qa.shape != qb.shape:
            failures.append(f"quality shape differs ({qa.shape} vs {qb.shape})")
        elif not np.array_equal(qa, qb):
            n_diff = int((qa != qb).sum())
            failures.append(
                f"quality vector differs in {n_diff}/{qa.size} positions "
                f"(means {qa.mean():.6f} vs {qb.mean():.6f})"
            )

    # if texts differ, show the first few so the cause is visible.
    # ascii() so a console with a narrow codepage cannot raise here -- this block
    # runs only on failure, and crashing would swallow the verdict below.
    if A["texts"] and B["texts"] and ha != hb:
        print("\n  first differing generations:")
        shown = 0
        for i, (x, y) in enumerate(zip(A["texts"], B["texts"])):
            if x != y:
                print(f"    [{i}] {args.label_a}: {ascii(x)}")
                print(f"    [{i}] {args.label_b}: {ascii(y)}")
                shown += 1
                if shown >= 3:
                    break

    print("\n" + "=" * 70)
    if failures:
        print("VERDICT: NOT COMPARABLE")
        for f in failures:
            print(f"  - {f}")
        print(
            "\nThe two tables are NOT about the same generations, so their PRR values\n"
            "cannot be placed side by side. Most likely cause: the runs resolved\n"
            "different stat calculators (attention capture on in one, off in the\n"
            "other), changing kernel selection under fp16. Fix by making the two\n"
            "configs agree on output_attentions, then re-run."
        )
        print("=" * 70)
        return 1

    print("VERDICT: COMPARABLE -- generations and quality are identical.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
