#!/usr/bin/env python3
"""Measure the ACTUAL 5-shot prompt length, with the real tokenizer and real data.

The attention-memory estimates in derive_tiers.py scale with L^2, so an assumed
prompt length is the weakest link in that argument. This builds the prompts
exactly as the eval pipeline does (same Dataset loader, same n_shot, same split)
and tokenizes them with the model's own tokenizer.

Downloads the dataset and the tokenizer only -- no model weights, so it runs on
CPU in under a minute.

Usage:
    python harness/measure_prompt_len.py [--n 200] [--config <cfg>]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("configs/stage1/eval_triviaqa_qwen.yaml"))
    ap.add_argument("--n", type=int, default=200,
                    help="How many prompts to measure.")
    ap.add_argument("--out", type=Path, default=Path("harness/prompt_length.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import resolve

    info = resolve(args.config)
    model_path = info["model_path"]
    dataset = info["dataset"]
    n_shot = info["n_shot"]
    split = info["eval_split"]

    print(f"[len] model     : {model_path}")
    print(f"[len] dataset   : {dataset}  split={split}  n_shot={n_shot}")

    from transformers import AutoTokenizer
    from lm_polygraph.utils.dataset import Dataset

    tok = AutoTokenizer.from_pretrained(model_path)

    data = Dataset.load(
        dataset,
        info_text_column(args.config),
        info_label_column(args.config),
        batch_size=1,
        prompt="",
        description="",
        n_shot=n_shot,
        few_shot_split="train",
        split=split,
        load_from_disk=False,
        trust_remote_code=False,
    )

    xs = data.x[: args.n]
    print(f"[len] measuring {len(xs)} prompts ...")

    lengths = [len(tok(x, add_special_tokens=True)["input_ids"]) for x in xs]
    arr = np.array(lengths)

    stats = {
        "model": model_path,
        "dataset": dataset,
        "n_shot": n_shot,
        "n_measured": int(arr.size),
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }

    print("\n=== 5-shot prompt length (tokens) ===")
    for k in ("min", "p50", "mean", "p95", "p99", "max"):
        print(f"  {k:5s} {stats[k]:8.1f}")

    print("\n--- example prompt (truncated) ---")
    print(xs[0][:400].replace("\n", "\\n"))
    print("---")

    args.out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n[len] wrote {args.out}")
    return stats


def _cfg_field(config, field, default):
    import yaml

    try:
        c = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
        return c.get(field, default)
    except Exception:
        return default


def info_text_column(config):
    return _cfg_field(config, "text_column", "input")


def info_label_column(config):
    return _cfg_field(config, "label_column", "output")


if __name__ == "__main__":
    main()
