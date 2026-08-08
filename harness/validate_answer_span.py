#!/usr/bin/env python3
"""Validate the 'generation == short answer' assumption with numbers, not assertion.

The ablation control defines the answer window as the whole generation. That is
only defensible if the generation actually IS a short answer. This measures it on
real generations from a completed run:

  * fraction that are a single line
  * token-length distribution of the generation
  * fraction whose normalised form matches the normalised gold answer exactly
  * fraction that are empty or hit the max_new_tokens ceiling

Run on the n=150 sign-check output BEFORE anything goes in the report, and again
per dataset -- the assumption may not transfer from TriviaQA to CoQA.

Reads the manager (for gold targets) and generations_<seed>.jsonl, so it needs no
GPU and no model.

Usage:
    python harness/validate_answer_span.py --save-dir <run dir> [--max-new-tokens 20]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_normalizer(config: Path):
    """Use the SAME normalizer the eval pipeline applies, resolved from the config."""
    try:
        import yaml

        cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        spec = cfg.get("process_output_fn")
        if not spec:
            return None, "none declared in config"
        path = config.parent / spec["path"]
        import importlib.util

        s = importlib.util.spec_from_file_location("norm_mod", path)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        return getattr(m, spec["fn_name"]), f"{spec['path']}:{spec['fn_name']}"
    except Exception as e:
        return None, f"unavailable ({e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=Path("configs/stage1/eval_triviaqa_qwen.yaml"))
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--show", type=int, default=10, help="Example generations to print.")
    args = ap.parse_args()

    import glob
    import torch

    mans = sorted(glob.glob(str(args.save_dir / "ue_manager_seed*")))
    if not mans:
        sys.exit(f"no ue_manager_seed* in {args.save_dir}")
    blob = torch.load(mans[0], weights_only=False)
    stats = (blob.get("stats") if isinstance(blob, dict) else blob.stats) or {}

    texts = stats.get("greedy_texts")
    tokens = stats.get("greedy_tokens")
    if not texts:
        sys.exit("greedy_texts absent from the manager; cannot validate.")

    norm, norm_src = load_normalizer(args.config)
    print(f"[span] normalizer: {norm_src}")
    print(f"[span] n generations: {len(texts)}")

    n = len(texts)
    single_line = sum(1 for t in texts if "\n" not in t.strip())
    empty = sum(1 for t in texts if not t.strip())
    lens = np.array([len(tk) for tk in tokens]) if tokens else np.array([])
    at_ceiling = int((lens >= args.max_new_tokens).sum()) if lens.size else 0

    print("\n=== is the generation a short answer? ===")
    print(f"  single line              : {single_line}/{n} = {single_line/n:.1%}")
    print(f"  empty                    : {empty}/{n} = {empty/n:.1%}")
    if lens.size:
        print(f"  gen tokens min/p50/p95/max: "
              f"{lens.min()}/{np.percentile(lens,50):.0f}/{np.percentile(lens,95):.0f}/{lens.max()}")
        print(f"  at max_new_tokens ceiling: {at_ceiling}/{n} = {at_ceiling/n:.1%}"
              + ("   <-- TRUNCATED: generation is NOT a complete short answer"
                 if at_ceiling / n > 0.10 else ""))

    # exact match against gold, using the pipeline's own normalizer
    gen_metrics = (blob.get("gen_metrics") if isinstance(blob, dict) else blob.gen_metrics) or {}
    acc = None
    for k, v in gen_metrics.items():
        name = k[1] if isinstance(k, tuple) and len(k) == 2 else str(k)
        if "Accuracy" in name:
            acc = np.asarray(v, dtype=np.float64)
            break
    if acc is not None:
        print(f"  exact match vs gold      : {acc.mean():.1%}  (n={acc.size})")
    else:
        print("  exact match vs gold      : no Accuracy metric in manager")

    print(f"\n=== first {args.show} generations (repr) ===")
    for t in texts[: args.show]:
        shown = t if len(t) < 90 else t[:90] + "..."
        print(f"  {shown!r}")

    verdict_ok = (single_line / n > 0.90) and (at_ceiling / n < 0.10) and (empty / n < 0.05)
    print("\n=== verdict ===")
    if verdict_ok:
        print("  OK: generations behave as short single-line answers;")
        print("      'answer window == generation' is defensible for this dataset.")
    else:
        print("  NOT OK: the generation is not reliably a short answer here.")
        print("      Do not claim 'span == answer' for this dataset without qualification.")

    out = args.save_dir / "answer_span_validation.json"
    out.write_text(json.dumps({
        "n": n,
        "single_line_frac": single_line / n,
        "empty_frac": empty / n,
        "at_ceiling_frac": at_ceiling / n if lens.size else None,
        "gen_len_p50": float(np.percentile(lens, 50)) if lens.size else None,
        "gen_len_p95": float(np.percentile(lens, 95)) if lens.size else None,
        "gen_len_max": int(lens.max()) if lens.size else None,
        "exact_match": float(acc.mean()) if acc is not None else None,
        "verdict_ok": bool(verdict_ok),
    }, indent=2), encoding="utf-8")
    print(f"\n[span] wrote {out}")
    return 0 if verdict_ok else 2


if __name__ == "__main__":
    sys.exit(main())
