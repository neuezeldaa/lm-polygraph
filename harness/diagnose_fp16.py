#!/usr/bin/env python3
"""Test the fp16-overflow hypothesis directly, on the prompts that actually failed.

Hypothesis: Qwen2.5 was trained in bf16; the T4 is Turing and has no hardware
bf16, so fp16 is the only half-precision option and its dynamic range is far
narrower. On some inputs activations exceed fp16 range, logits go inf/NaN, and
argmax collapses to token id 0 -- which decodes to '!' in Qwen's vocabulary,
producing the observed '!!!!!!!!!!' generations.

This runs the SAME prompts under fp16 and float32 and compares:
  * are the raw logits finite?
  * what is the largest magnitude reached, versus fp16's max of 65504?
  * does the generation become sane?

Selects prompts automatically from a completed run: the ones whose generation
collapsed to '!' (or another degenerate repeat). Falls back to the first N
prompts if no run directory is given.

Weights are loaded ONCE per dtype. float32 for a 3B model is ~12.4 GB, which is
tight on a 16 GB T4, so the two passes are run sequentially with the first model
freed in between.

Usage:
    python harness/diagnose_fp16.py --save-dir <n150 run dir> --n 10
    python harness/diagnose_fp16.py --n 10          # without a prior run
"""

import argparse
import gc
import glob
import json
import sys
from pathlib import Path

FP16_MAX = 65504.0


def pick_prompts(save_dir, n, config):
    """Prefer prompts whose generation collapsed; else take the first n."""
    import torch

    texts = None
    if save_dir:
        mans = sorted(glob.glob(str(Path(save_dir) / "ue_manager_seed*")))
        if mans:
            blob = torch.load(mans[0], weights_only=False)
            stats = (blob.get("stats") if isinstance(blob, dict) else blob.stats) or {}
            texts = stats.get("greedy_texts")

    from harness.provenance import resolve
    from lm_polygraph.utils.dataset import Dataset
    import yaml

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    info = resolve(Path(config))
    data = Dataset.load(
        info["dataset"], cfg.get("text_column", "input"), cfg.get("label_column", "output"),
        batch_size=1, prompt="", description="", n_shot=info["n_shot"],
        few_shot_split="train", split=info["eval_split"],
        load_from_disk=False, trust_remote_code=False,
    )
    xs = list(data.x)

    if texts:
        bad = [i for i, t in enumerate(texts)
               if t.strip() and len(set(t.strip())) <= 2 and len(t.strip()) >= 5]
        if bad:
            print(f"[fp16] {len(bad)} collapsed generations found; using the first {n}")
            idx = bad[:n]
            return [xs[i] for i in idx if i < len(xs)], idx
        print("[fp16] no collapsed generations found in that run; using the first prompts")
    return xs[:n], list(range(min(n, len(xs))))


def run_dtype(model_path, dtype, prompts, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = str(dtype).replace("torch.", "")
    print(f"\n{'='*70}\n{name}\n{'='*70}")

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="eager", device_map="cuda"
    )
    model.eval()
    print(f"  weights on GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    results = []
    for i, p in enumerate(prompts):
        batch = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**batch)
            logits = out.logits[0, -1, :].float()

            gen = model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(gen[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)

        finite = bool(torch.isfinite(logits).all())
        amax = float(logits.abs().max()) if finite else float("inf")
        top = int(logits.argmax())
        results.append({
            "idx": i, "finite": finite, "abs_max_logit": amax,
            "argmax_token": top, "argmax_decoded": tok.decode([top]),
            "generation": text,
        })
        flag = "" if finite else "   <-- NON-FINITE"
        print(f"  [{i}] finite={finite}  |logit|max={amax:9.1f}  "
              f"argmax={top:6d} {tok.decode([top])!r:12s}{flag}")
        print(f"       gen: {ascii(text[:70])}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default=None, help="A completed run, to pick failing prompts from.")
    ap.add_argument("--config", default="configs/stage1/eval_triviaqa_qwen.yaml")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--out", default="harness/fp16_diagnosis.json")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch

    assert torch.cuda.is_available(), "needs a GPU"
    print("device:", torch.cuda.get_device_name(0))
    print(f"fp16 max representable magnitude: {FP16_MAX}")

    from harness.provenance import resolve
    model_path = resolve(Path(args.config))["model_path"]
    print("model :", model_path)

    prompts, idx = pick_prompts(args.save_dir, args.n, args.config)
    print(f"[fp16] testing {len(prompts)} prompts (dataset indices {idx})")

    fp16 = run_dtype(model_path, torch.float16, prompts, args.max_new_tokens)
    fp32 = run_dtype(model_path, torch.float32, prompts, args.max_new_tokens)

    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    n = len(fp16)
    fp16_bad = sum(1 for r in fp16 if not r["finite"])
    fp32_bad = sum(1 for r in fp32 if not r["finite"])
    changed = sum(1 for a, b in zip(fp16, fp32) if a["generation"] != b["generation"])
    near = sum(1 for r in fp32 if r["finite"] and r["abs_max_logit"] > FP16_MAX * 0.5)

    print(f"  non-finite logits   fp16: {fp16_bad}/{n}   float32: {fp32_bad}/{n}")
    print(f"  generations differing between dtypes: {changed}/{n}")
    print(f"  float32 |logit| exceeding half of fp16's max: {near}/{n}")
    if fp32:
        print(f"  largest float32 |logit| seen: {max(r['abs_max_logit'] for r in fp32):.1f}"
              f"  (fp16 max {FP16_MAX})")

    if fp16_bad and not fp32_bad:
        print("\n  CONFIRMED: fp16 produces non-finite logits where float32 does not.")
    elif not fp16_bad and not fp32_bad:
        print("\n  NOT REPRODUCED on these prompts: logits were finite in both dtypes.")
        print("  The collapse may be input-specific -- rerun with --save-dir pointing")
        print("  at the failing run so the actual failing prompts are selected.")
    else:
        print("\n  INCONCLUSIVE: see the per-prompt table above.")

    Path(args.out).write_text(json.dumps(
        {"model": model_path, "indices": idx, "fp16": fp16, "float32": fp32}, indent=2
    ), encoding="utf-8")
    print(f"\n[fp16] wrote {args.out}")


if __name__ == "__main__":
    main()
