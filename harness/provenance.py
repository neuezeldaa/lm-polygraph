#!/usr/bin/env python3
"""Resolve and print the model identity a config ACTUALLY composes to.

Model provenance must never be ambiguous: this reads the Hydra config exactly the
way ``polygraph_eval`` does, resolves the loader script, and reports the model
path, dtype and device that will really be used -- not what any document claims.

Used two ways:
  * as Colab cell 1:  python harness/provenance.py --config <cfg>
  * from run_baselines.py preflight, so every run prints it

``--expect`` turns it into a hard assertion, so a silent model swap fails the run
instead of producing numbers with unclear provenance.
"""

import argparse
import ast
import re
import sys
from pathlib import Path


def resolve(config: Path) -> dict:
    from hydra import initialize_config_dir, compose

    config = config.resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config.parent)):
        cfg = compose(config_name=config.stem)

    info = {
        "config": str(config),
        "model_path": cfg.model.path,
        "model_type": getattr(cfg.model, "type", None),
        "device_map": dict(getattr(cfg.model, "load_model_args", {}) or {}).get(
            "device_map"
        ),
        "load_script": getattr(cfg.model, "path_to_load_script", None),
        "add_bos_token": getattr(cfg.model, "add_bos_token", None),
        "max_new_tokens": getattr(cfg, "max_new_tokens", None),
        "dataset": list(cfg.dataset) if getattr(cfg, "dataset", None) else None,
        "eval_split": getattr(cfg, "eval_split", None),
        "n_shot": getattr(cfg, "n_shot", None),
        "subsample": getattr(cfg, "subsample_eval_dataset", None),
        "seed": list(cfg.seed) if getattr(cfg, "seed", None) else None,
        "batch_size": getattr(cfg, "batch_size", None),
        "n_estimators": len(cfg.estimators),
        "generation_params": dict(getattr(cfg, "generation_params", {}) or {}),
    }

    # dtype / attention are pinned inside the loader script, not the yaml
    dtype = attn = None
    if info["load_script"]:
        script = config.parent / info["load_script"]
        if script.exists():
            src = script.read_text(encoding="utf-8")
            try:  # prefer the parsed call kwargs over a regex on comments
                tree = ast.parse(src)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        for kw in node.keywords or []:
                            if kw.arg == "torch_dtype":
                                dtype = ast.unparse(kw.value)
                            elif kw.arg == "attn_implementation":
                                dtype_v = ast.literal_eval(kw.value) if isinstance(
                                    kw.value, ast.Constant
                                ) else ast.unparse(kw.value)
                                attn = dtype_v
            except SyntaxError:
                m = re.search(r"torch_dtype\s*=\s*([\w.]+)", src)
                dtype = m.group(1) if m else None
    info["dtype"] = dtype
    info["attn_implementation"] = attn
    return info


def runtime_device() -> dict:
    out = {}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["device_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            out["vram_free_gb"] = round(free / 1e9, 2)
            out["vram_total_gb"] = round(total / 1e9, 2)
        else:
            out["device_name"] = "CPU (no CUDA)"
    except Exception as e:
        out["error"] = repr(e)
    return out


def report(config: Path, expect: str = None) -> dict:
    info = resolve(config)
    rt = runtime_device()

    print("=" * 68)
    print("MODEL PROVENANCE  (resolved from the config, not from any document)")
    print("=" * 68)
    print(f"  config           : {info['config']}")
    print(f"  model.path       : {info['model_path']}")
    print(f"  model.type       : {info['model_type']}")
    print(f"  dtype            : {info['dtype']}")
    print(f"  attn_impl        : {info['attn_implementation']}")
    print(f"  device_map       : {info['device_map']}")
    print(f"  loader script    : {info['load_script']}")
    print("-" * 68)
    print(f"  runtime device   : {rt.get('device_name')}")
    if "vram_total_gb" in rt:
        print(f"  VRAM             : {rt['vram_free_gb']} GB free / {rt['vram_total_gb']} GB")
    print(f"  torch            : {rt.get('torch')}")
    print("-" * 68)
    print(f"  dataset          : {info['dataset']}  split={info['eval_split']}  n_shot={info['n_shot']}")
    print(f"  generation       : {info['generation_params']}  max_new_tokens={info['max_new_tokens']}")
    print(f"  n (subsample)    : {info['subsample']}   batch_size={info['batch_size']}   seed={info['seed']}")
    print(f"  n estimators     : {info['n_estimators']}")
    print("-" * 68)
    # fp32_projection changes the method's numerics, so it is REPORTED, not a
    # silent default. "dE needs an fp32 projection to be stable on a T4" is part
    # of the finding, and a reader has to know the numbers were produced with it.
    try:
        from lm_polygraph.stat_calculators.energy import EnergyCalculator

        ec = EnergyCalculator()
        print(f"  EnergyCalculator : fp32_projection={ec.fp32_projection}, "
              f"vocab_chunk={ec.vocab_chunk}")
        if ec.fp32_projection:
            print("                     (vocabulary projection redone in float32 for")
            print("                      the scored rows; dE amplifies logit error ~6.5x)")
    except Exception as e:
        print(f"  EnergyCalculator : could not introspect ({e})")
    print("=" * 68)

    if expect and info["model_path"] != expect:
        sys.exit(
            f"\nFATAL: model provenance mismatch.\n"
            f"  expected : {expect}\n"
            f"  resolved : {info['model_path']}\n"
            f"Refusing to run: numbers whose model provenance is ambiguous are not usable."
        )
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("configs/stage1/eval_triviaqa_qwen.yaml"))
    ap.add_argument("--expect", default=None,
                    help="Fail unless model.path equals this exactly.")
    args = ap.parse_args()
    report(args.config, args.expect)


if __name__ == "__main__":
    main()
