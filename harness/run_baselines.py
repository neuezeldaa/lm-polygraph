#!/usr/bin/env python3
"""Runner + offline analysis for the Spilled Energy experiments.

Two first-class entry points:

  1. RUN (needs a GPU + the model):
         python harness/run_baselines.py --config <cfg> --save-dir <dir>
     Invokes lm-polygraph's ``polygraph_eval`` with a frozen Hydra config, then
     analyses the result.

  2. ANALYSE OFFLINE (CPU, no model, no network):
         python harness/run_baselines.py --skip-run --save-dir <dir>
     Parses an existing ``ue_manager_seed*`` file, applies the accuracy gate,
     writes the per-sample arrays, and builds the PRR@0.5 table with bootstrap
     confidence intervals. This is the path used to analyse a manager downloaded
     from Colab.

Outputs, all written next to the manager:
  * ``per_sample_<seed>.npz``   -- per-sample estimator + quality vectors (for CIs)
  * ``generations_<seed>.jsonl`` -- cached greedy generations
  * ``prr_0.5_table.md``        -- the reported table

Hard-fails (nonzero exit) if the quality function's mean falls outside
[--acc-min, --acc-max]: at 4% or 95% accuracy the PRR ranking is noise and every
downstream number is meaningless.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PRR_KEY = "prr_0.5_normalized"
PRR_RAW_KEY = "prr_0.5"


# ---------------------------------------------------------------------------
# manager access helpers (tolerate dict OR pickled UEManager object)
# ---------------------------------------------------------------------------
def _field(man, name):
    if isinstance(man, dict):
        return man.get(name)
    return getattr(man, name, None)


def load_manager(path):
    import torch

    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# PRR, recomputed locally so bootstrap resamples are possible
# ---------------------------------------------------------------------------
def _normalize_target(target):
    t = np.asarray(target, dtype=np.float64)
    lo, hi = np.min(t), np.max(t)
    if np.isclose(lo, hi):
        lo, hi = lo - 1, hi + 1
    return (t - lo) / (hi - lo)


def prr(ue, target, max_rejection=0.5):
    """Area under the prediction-rejection curve (mirrors lm_polygraph's impl)."""
    target = _normalize_target(target)
    ue = np.asarray(ue, dtype=np.float64)
    num_obs = len(ue)
    num_rej = int(max_rejection * num_obs)
    if num_rej < 1:
        return np.nan
    order = np.argsort(ue)
    sorted_metrics = np.asarray(target)[order]
    cumsum = np.cumsum(sorted_metrics)[-num_rej:]
    scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
    return float(np.sum(scores) / num_rej)


def prr_normalized(ue, target, max_rejection=0.5, n_random=200, seed=42):
    """(prr - random) / (oracle - random), matching UEManager.eval_ue()."""
    t = np.asarray(target, dtype=np.float64)
    val = prr(ue, t, max_rejection)
    oracle = prr(-t, t, max_rejection)
    rng = np.random.default_rng(seed)
    rand_scores = np.arange(len(t), dtype=np.float64)
    vals = []
    for _ in range(n_random):
        rng.shuffle(rand_scores)
        vals.append(prr(rand_scores, t, max_rejection))
    random = float(np.mean(vals))
    if oracle == random:
        return val
    return (val - random) / (oracle - random)


def bootstrap_ci(ue, target, n_boot=1000, seed=0, max_rejection=0.5):
    """Percentile bootstrap CI for normalized PRR. Resamples sample indices."""
    ue = np.asarray(ue, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = len(ue)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t = target[idx]
        if np.isclose(np.min(t), np.max(t)):
            continue  # degenerate resample: all-correct or all-wrong
        vals.append(prr_normalized(ue[idx], t, max_rejection, n_random=30, seed=42))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ---------------------------------------------------------------------------
def run_eval(config, save_dir, samples, seed, extra_overrides):
    save_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HYDRA_CONFIG"] = str(config)
    # let the subprocess import harness.* estimators referenced by dotted path
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    overrides = [f"save_path={save_dir.as_posix()}"]
    if samples is not None:
        overrides.append(f"subsample_eval_dataset={samples}")
    if seed is not None:
        overrides.append(f"seed=[{seed}]")
    overrides += list(extra_overrides)

    cmd = ["polygraph_eval", *overrides]
    print(f"[run] HYDRA_CONFIG={config}")
    print(f"[run] {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env)
    except FileNotFoundError:
        sys.exit(
            "[run] FATAL: 'polygraph_eval' is not on PATH.\n"
            "  This happens on Colab right after `pip install -e .` when the script\n"
            "  dir is not exported. Fix with either:\n"
            "    export PATH=\"$PATH:$(python -c 'import sysconfig;print(sysconfig.get_path(\"scripts\"))')\"\n"
            "  or run the module directly:\n"
            "    HYDRA_CONFIG=<cfg> python <repo>/scripts/polygraph_eval"
        )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        sys.exit(f"[run] polygraph_eval failed with code {proc.returncode}")

    # Record wall-clock so a long run can be projected from a short one BEFORE
    # committing to it (harness/estimate_runtime.py).
    meta = {
        "elapsed_sec": round(elapsed, 1),
        "n": samples,
        "config": str(config),
        "save_dir": str(save_dir),
    }
    (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[run] wall clock {elapsed/60:.1f} min  ->  {save_dir / 'run_meta.json'}")


def preflight_config(config: Path, expect_model: str = None):
    """Print model provenance and verify the config saves generations,
    BEFORE burning a long run."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from provenance import report

        report(config, expect_model)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[preflight] WARNING: could not resolve model provenance: {e}")

    try:
        import yaml

        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[preflight] WARNING: could not parse {config}: {e}")
        return
    save_stats = cfg.get("save_stats") or []
    missing = [k for k in ("greedy_texts", "greedy_tokens") if k not in save_stats]
    if missing:
        print(
            f"[preflight] WARNING: {config.name} save_stats is missing {missing}; "
            "the generation cache will be incomplete."
        )
    else:
        print(f"[preflight] OK: save_stats includes {save_stats}")


# ---------------------------------------------------------------------------
def collect(man):
    """Returns (prr_rows, estimations, gen_metrics, diagnostics)."""
    metrics = _field(man, "metrics") or {}
    estimations = _field(man, "estimations") or {}
    gen_metrics = _field(man, "gen_metrics") or {}

    rows, ue_names_seen, example_key = {}, set(), None
    for key, val in metrics.items():
        if example_key is None:
            example_key = key
        if not (isinstance(key, tuple) and len(key) == 4):
            continue
        _lvl, e_name, gen_name, ue_name = key
        ue_names_seen.add(ue_name)
        if ue_name not in (PRR_KEY, PRR_RAW_KEY):
            continue
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            rows.setdefault((e_name, gen_name), {})[ue_name] = None
            continue
        try:
            rows.setdefault((e_name, gen_name), {})[ue_name] = float(val)
        except (TypeError, ValueError):
            rows.setdefault((e_name, gen_name), {})[ue_name] = None

    diag = {
        "ue_names_seen": sorted(ue_names_seen),
        "example_key": example_key,
        "n_metrics": len(metrics),
    }
    return rows, estimations, gen_metrics, diag


def accuracy_gate(gen_metrics, lo, hi):
    """Print the mean of every quality function; hard-fail if out of band."""
    print("\n=== quality function means (accuracy gate) ===")
    if not gen_metrics:
        sys.exit("[gate] FATAL: no gen_metrics in the manager; cannot validate the run.")

    failures = []
    for key, vals in gen_metrics.items():
        name = key[1] if isinstance(key, tuple) and len(key) == 2 else str(key)
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            print(f"  {name:30s} EMPTY")
            continue
        m = float(arr.mean())
        # Only band-check metrics that look like a 0/1 correctness score.
        bounded = float(arr.min()) >= -1e-9 and float(arr.max()) <= 1 + 1e-9
        flag = ""
        if bounded and not (lo <= m <= hi):
            flag = f"  <-- OUT OF BAND [{lo}, {hi}]"
            failures.append((name, m))
        print(f"  {name:30s} mean={m:.4f}  n={arr.size}{flag}")

    if failures:
        det = ", ".join(f"{n}={m:.3f}" for n, m in failures)
        sys.exit(
            f"\n[gate] FATAL: quality out of band ({det}).\n"
            "  PRR is meaningless at this accuracy. Likely cause: prompt format\n"
            "  (e.g. an instruct model fed a plain continuation few-shot prompt\n"
            "  instead of its chat template), or over-aggressive answer normalization.\n"
            "  Inspect generations_<seed>.jsonl before trusting any number.\n"
            "  Override with --acc-min/--acc-max only if you know why."
        )
    print("[gate] PASS")


def dump_per_sample(estimations, gen_metrics, path):
    """Persist per-sample vectors so CIs can be recomputed offline."""
    arrays, meta = {}, {"estimators": [], "gen_metrics": []}
    for key, vals in estimations.items():
        name = key[1] if isinstance(key, tuple) and len(key) == 2 else str(key)
        arrays[f"ue::{name}"] = np.asarray(vals, dtype=np.float64)
        meta["estimators"].append(name)
    for key, vals in gen_metrics.items():
        name = key[1] if isinstance(key, tuple) and len(key) == 2 else str(key)
        arrays[f"gm::{name}"] = np.asarray(vals, dtype=np.float64)
        meta["gen_metrics"].append(name)
    if not arrays:
        print("[dump] WARNING: nothing to persist")
        return None
    np.savez_compressed(path, **arrays)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[dump] per-sample arrays -> {path}  ({len(meta['estimators'])} estimators)")
    return arrays


def build_table(rows, estimations, gen_metrics, n, n_boot, gen_name_hint=None):
    gen_names = sorted({g for (_e, g) in rows})
    if gen_name_hint and gen_name_hint in gen_names:
        gen_names = [gen_name_hint]

    est_by_name = {
        (k[1] if isinstance(k, tuple) and len(k) == 2 else str(k)): np.asarray(v, float)
        for k, v in estimations.items()
    }
    gm_by_name = {
        (k[1] if isinstance(k, tuple) and len(k) == 2 else str(k)): np.asarray(v, float)
        for k, v in gen_metrics.items()
    }

    out = []
    for gen in gen_names:
        sub = {e: d for (e, g), d in rows.items() if g == gen}
        target = gm_by_name.get(gen)

        enriched = []
        for e, d in sub.items():
            norm = d.get(PRR_KEY)
            raw = d.get(PRR_RAW_KEY)
            lo = hi = None
            ue = est_by_name.get(e)
            if n_boot > 0 and ue is not None and target is not None and len(ue) == len(target):
                mask = np.isfinite(ue) & np.isfinite(target)
                if mask.sum() > 10:
                    lo, hi = bootstrap_ci(ue[mask], target[mask], n_boot=n_boot)
            enriched.append((e, norm, raw, lo, hi))

        enriched.sort(key=lambda r: (r[1] if r[1] is not None else -np.inf), reverse=True)

        out.append(f"### Normalized PRR@0.5 — quality: `{gen}` (n={n})\n")
        ci_hdr = f" | 95% CI ({n_boot} boot)" if n_boot > 0 else ""
        out.append(f"| Estimator | normalized PRR@0.5{ci_hdr} | raw PRR@0.5 |")
        out.append("|---|---:|" + ("---:|" if n_boot > 0 else "") + "---:|")
        for e, norm, raw, lo, hi in enriched:
            ns = f"{norm:.4f}" if norm is not None else "—"
            rs = f"{raw:.4f}" if raw is not None else "—"
            if n_boot > 0:
                cs = f"[{lo:.3f}, {hi:.3f}]" if lo is not None and np.isfinite(lo) else "—"
                out.append(f"| {e} | {ns} | {cs} | {rs} |")
            else:
                out.append(f"| {e} | {ns} | {rs} |")
        out.append("")
    return "\n".join(out)


def dump_generations(man, path):
    stats = _field(man, "stats") or {}
    texts = stats.get("greedy_texts")
    if not texts:
        print("[cache] WARNING: greedy_texts absent from stats; no generation cache.")
        return
    tokens = stats.get("greedy_tokens")
    with open(path, "w", encoding="utf-8") as fh:
        for i, t in enumerate(texts):
            row = {"idx": i, "greedy_text": t}
            if tokens is not None and i < len(tokens):
                row["greedy_tokens"] = list(tokens[i])
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[cache] {len(texts)} generations -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--skip-run", action="store_true",
                    help="Analyse an existing manager; no GPU/model needed.")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Bootstrap resamples for the CI (0 disables).")
    ap.add_argument("--acc-min", type=float, default=0.10)
    ap.add_argument("--acc-max", type=float, default=0.90)
    ap.add_argument("--expect-model", default=None,
                    help="Fail unless the config resolves to this model.path.")
    ap.add_argument("--quality", default=None,
                    help="Name of the quality function to report against "
                         "(default: all present).")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    if not args.skip_run:
        if args.config is None:
            sys.exit("--config is required unless --skip-run is given")
        cfg = args.config.resolve()
        if not cfg.exists():
            sys.exit(f"config not found: {cfg}")
        preflight_config(cfg, args.expect_model)
        run_eval(cfg, args.save_dir, args.samples, args.seed, args.overrides)

    files = sorted(glob.glob(str(args.save_dir / "ue_manager_seed*")))
    if not files:
        sys.exit(f"no ue_manager_seed* found in {args.save_dir}")

    all_md = []
    for path in files:
        tag = Path(path).name.replace("ue_manager_", "")
        print(f"\n{'='*70}\n[analyse] {path}\n{'='*70}")
        man = load_manager(path)
        rows, estimations, gen_metrics, diag = collect(man)

        if not rows:
            print("[analyse] FATAL: no PRR metrics parsed.")
            print(f"  metrics entries      : {diag['n_metrics']}")
            print(f"  distinct ue_metric   : {diag['ue_names_seen']}")
            print(f"  example key          : {diag['example_key']!r}")
            print(f"  expected one of      : {[PRR_KEY, PRR_RAW_KEY]}")
            sys.exit(1)

        accuracy_gate(gen_metrics, args.acc_min, args.acc_max)

        n = None
        for v in estimations.values():
            n = len(v)
            break

        dump_per_sample(estimations, gen_metrics, args.save_dir / f"per_sample_{tag}.npz")
        dump_generations(man, args.save_dir / f"generations_{tag}.jsonl")

        table = build_table(rows, estimations, gen_metrics, n, args.n_boot, args.quality)
        print("\n" + table)
        all_md.append(f"## {tag}\n\n{table}")

    out_md = args.save_dir / "prr_0.5_table.md"
    out_md.write_text("\n".join(all_md), encoding="utf-8")
    print(f"[analyse] table -> {out_md}")


if __name__ == "__main__":
    main()
