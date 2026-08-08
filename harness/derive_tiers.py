#!/usr/bin/env python3
"""Derive the feasible baseline set MECHANICALLY from what lm-polygraph ships.

The primary baseline set must not look hand-picked. This script takes upstream's
``examples/configs/estimators/default_estimators.yaml`` as the population, then
classifies every row by predicates evaluated over its *resolved transitive
dependency set* -- never over a hardcoded list of estimator names.

Pipeline per row (a row is one name+cfg pair; duplicate names with different cfg
are distinct rows, e.g. the four LexicalSimilarity metrics):

  1. instantiate through FactoryEstimator (the same path polygraph_eval uses)
  2. read ``stats_dependencies``
  3. resolve those stat names transitively to the StatCalculators that would
     actually be constructed, using BOTH registration sources:
       * register_default_stat_calculators("Whitebox")  (python defaults)
       * the upstream stat_calculators YAML group        (e.g. the training
         statistics calculator, which is only registered there)
  4. evaluate the predicates below over the resolved stats + calculators + cfg

Predicates (all structural):
  needs_train_data      resolved stat name matches ^(train_|background_), or a
                        resolved calculator cfg carries train/background split keys
  needs_external_corpus a cfg value looks like an external dataset/corpus id
  needs_sampling        resolved stat name or calculator name indicates sampled
                        generations
  needs_auxiliary_model a resolved calculator cfg names a second neural model
                        (NLI / cross-encoder / deberta / sentence-transformer)
  needs_attention       depends on stored attention maps (flagged separately: the
                        cost is in generating with output_attentions=True)

Tier = first matching predicate in the priority order above; otherwise
``single_pass_cheap``. All raw flags are emitted too, so a row tripping several
predicates stays auditable.

Usage:
    python harness/derive_tiers.py [--estimators <yaml>] [--out-dir harness]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ESTIMATORS = REPO / "examples" / "configs" / "estimators" / "default_estimators.yaml"
DEFAULT_CALC_GROUP = REPO / "examples" / "configs" / "stat_calculators" / "default_calculators.yaml"

TIER_PRIORITY = [
    "needs_train_data",
    "needs_external_corpus",
    "needs_sampling",
    "single_pass_plus_aux_model",
]

# Attention-memory model. Qwen2.5-3B-Instruct is 36 layers x 16 heads.
# ATTN_SEQ_LEN is MEASURED, not assumed: harness/measure_prompt_len.py tokenizes
# real 5-shot TriviaQA prompts with the model's own tokenizer and writes
# harness/prompt_length.json, which this script reads. Memory scales with L^2, so
# this is the value the whole feasibility argument hinges on.
ATTN_LAYERS = 36
ATTN_HEADS = 16
ATTN_SEQ_LEN = 185  # p99 of the measured distribution; overridden by the json
ATTN_GEN_LEN = 20
ATTN_BATCH = 4  # the frozen config's batch_size; overridable with --batch-size
# AttentionForwardPassCalculator does torch.cat(attentions).float().numpy(), i.e.
# [layers, heads, L, L] float32 per sample, then pads the batch to the longest
# sequence and copies again via np.array -- so peak ~ 2 x batch x per-sample.
# Budget is expressed per BATCH, because "is this feasible" depends on batch size:
# a method that fails at batch 4 may be fine at batch 1.
ATTN_BATCH_BUDGET_BYTES = 4 * 1024**3  # 4 GB of a free Colab VM's ~12.7 GB

EXCLUSION_REASON = {
    "needs_train_data": (
        "Fits statistics on a train/background split. Excluded: the assignment "
        "forbids supervised training, and this is fitting on held-out train/background data "
        "in all but name."
    ),
    "needs_external_corpus": (
        "Downloads a large external corpus or artifact at init. Excluded: not "
        "feasible on a free T4 session and not required by any constraint."
    ),
    "needs_sampling": (
        "Requires multiple sampled generations per input. Excluded from the "
        "primary table: cost is a multiple of the single-pass budget, so it is "
        "not a matched-compute comparison. Reported separately at n=300."
    ),
    "needs_auxiliary_model": (
        "Requires a second neural model (NLI / cross-encoder) resident on the "
        "GPU alongside the 3B LM. Excluded: VRAM pressure on a 16GB T4."
    ),
    "unsafe_attention_memory": (
        f"Materialises full-sequence attention tensors. At {ATTN_LAYERS} layers x "
        f"{ATTN_HEADS} heads over a ~{ATTN_SEQ_LEN}-token 5-shot prompt this is "
        "GBs per sample, stored as float32 on CPU. Excluded: unsafe on a T4 at "
        "this prompt length."
    ),
    "single_pass_plus_aux_model": (
        "One generation plus a second neural model (NLI cross-encoder) over that "
        "generation -- no sampling. INCLUDED in the primary table as its own row, "
        "flagged: it is materially cheaper than the sampling tier but is not a "
        "pure single-pass method."
    ),
    "single_pass_cheap": "",
    "instantiation_failed": "Could not be constructed; see error column.",
}

# --- structural cues -------------------------------------------------------
TRAIN_STAT_RE = re.compile(r"^(train_|background_)")
TRAIN_CFG_KEYS = {
    "train_split", "subsample_train_dataset", "background_train_dataset",
    "subsample_background_train_dataset", "background_load_from_disk",
}
SAMPLE_STAT_RE = re.compile(r"(^sample_|_sample_|^samples$|sample_texts|sample_log)")
SAMPLE_CALC_RE = re.compile(r"sampling", re.I)
AUX_MODEL_CFG_KEYS = {
    "nli_model", "deberta_path", "cross_encoder_name", "sentence_model",
    "similarity_model",
}
CORPUS_CFG_KEYS = {
    "idf_dataset", "background_train_dataset", "dataset", "corpus",
    "spacy_path",
}
ATTENTION_STAT_RE = re.compile(r"attention", re.I)
# `attention_all` is stored per generated token only ([L*H, gen, gen]); every other
# attention statistic in the repo is materialised over the FULL input sequence,
# which is what makes it expensive with a 5-shot prompt.
GEN_ONLY_ATTENTION_STATS = {"attention_all"}


def flatten_cfg(cfg, prefix=""):
    """Yield (dotted_key, value) for every scalar in a nested config.

    OmegaConf nodes are converted with resolve=False: these configs carry
    interpolations like ``${dataset}`` that reference a parent config which does
    not exist here, and resolving them raises. We want the literal text anyway --
    an unresolved ``${...}`` marks a value supplied by the eval config rather
    than an external corpus.
    """
    if cfg is None:
        return
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        pass

    if hasattr(cfg, "items"):
        for k, v in cfg.items():
            key = f"{prefix}{k}"
            if hasattr(v, "items") or isinstance(v, (list, tuple)):
                yield from flatten_cfg(v, prefix=f"{key}.")
            else:
                yield key, v
    elif isinstance(cfg, (list, tuple)):
        for i, v in enumerate(cfg):
            yield from flatten_cfg(v, prefix=f"{prefix}{i}.")


def build_stat_index():
    """stat name -> [calculator containers], from BOTH registration sources."""
    from lm_polygraph.defaults.register_default_stat_calculators import (
        register_default_stat_calculators,
    )
    from lm_polygraph.utils.factory_stat_calculator import StatCalculatorContainer
    from omegaconf import OmegaConf

    containers = list(
        register_default_stat_calculators(
            "Whitebox", output_attentions=True, output_hidden_states=True
        )
    )

    # the YAML group registers extra calculators (notably the training-statistics
    # one) that the python defaults do not
    if DEFAULT_CALC_GROUP.exists():
        entries = yaml.safe_load(DEFAULT_CALC_GROUP.read_text(encoding="utf-8")) or []
        for e in entries:
            if not isinstance(e, dict):
                continue  # the bare "auto" entry
            containers.append(
                StatCalculatorContainer(
                    name=e.get("name"),
                    stats=e.get("stats") or [],
                    dependencies=e.get("dependencies") or [],
                    builder=e.get("builder"),
                    cfg=OmegaConf.create(e.get("cfg") or {}),
                )
            )

    index = {}
    for c in containers:
        for s in c.stats or []:
            index.setdefault(s, []).append(c)
    return containers, index


def resolve_transitive(deps, index):
    """Return (all stat names, {calc name: container}) reachable from deps."""
    seen_stats, calcs, unresolved = set(), {}, set()
    frontier = list(deps)
    while frontier:
        s = frontier.pop()
        if s in seen_stats:
            continue
        seen_stats.add(s)
        producers = index.get(s)
        if not producers:
            unresolved.add(s)
            continue
        for c in producers:
            calcs[c.name] = c
            for d in c.dependencies or []:
                if d not in seen_stats:
                    frontier.append(d)
    return seen_stats, calcs, unresolved


def classify(stats, calcs, est_cfg):
    flags = set()

    cfg_pairs = list(flatten_cfg(est_cfg))
    for c in calcs.values():
        cfg_pairs += list(flatten_cfg(c.cfg))
    cfg_keys = {k.split(".")[-1] for k, _ in cfg_pairs}

    # train / background data
    if any(TRAIN_STAT_RE.search(s) for s in stats) or (cfg_keys & TRAIN_CFG_KEYS):
        flags.add("needs_train_data")

    # external corpus / artifact
    for k, v in cfg_pairs:
        leaf = k.split(".")[-1]
        if leaf in CORPUS_CFG_KEYS and isinstance(v, str) and v.strip():
            # a bare interpolation like ${dataset} is the eval set, not external
            if not v.startswith("${"):
                flags.add("needs_external_corpus")

    # sampling
    if any(SAMPLE_STAT_RE.search(s) for s in stats) or any(
        SAMPLE_CALC_RE.search(n) for n in calcs
    ):
        flags.add("needs_sampling")

    # auxiliary neural model
    if cfg_keys & AUX_MODEL_CFG_KEYS:
        flags.add("needs_auxiliary_model")
        # Without sampling, a second model is one extra encoder pass over the
        # generation -- far cheaper than the sampling tier, so it gets its own
        # tier and stays in the primary table rather than being dropped.
        if "needs_sampling" not in flags:
            flags.add("single_pass_plus_aux_model")

    # attention tensors: flag, and size them -- this is a real T4 constraint, not
    # a formality. AttentionForwardPassCalculator stores
    # torch.cat(attentions).float().numpy() over the whole sequence, per sample.
    attn_stats = {s for s in stats if ATTENTION_STAT_RE.search(s)}
    attn_bytes = 0
    for s in attn_stats:
        span = ATTN_GEN_LEN if s in GEN_ONLY_ATTENTION_STATS else ATTN_SEQ_LEN
        attn_bytes = max(attn_bytes, ATTN_LAYERS * ATTN_HEADS * span * span * 4)
    # peak is per batch, and the padding step copies it once more
    batch_peak = attn_bytes * ATTN_BATCH * 2
    if attn_stats:
        flags.add("needs_attention")
        if batch_peak > ATTN_BATCH_BUDGET_BYTES:
            flags.add("unsafe_attention_memory")

    tier = "single_pass_cheap"
    for t in TIER_PRIORITY:
        if t in flags:
            tier = t
            break
    if tier == "single_pass_cheap" and "unsafe_attention_memory" in flags:
        tier = "unsafe_attention_memory"
    return tier, flags, attn_bytes, batch_peak


def count_model_passes(calcs):
    """How many times each resolved calculator invokes the model.

    Derived by inspecting each calculator class's own source for generate/forward
    call sites, so it stays mechanical rather than a curated list. Used to qualify
    the 'matched single-pass budget' claim: PTrue and the PMI family each add a
    second pass, so they are ~2x the cost of MSP/Perplexity.
    """
    import inspect

    total, detail = 0, {}
    for name, c in sorted(calcs.items()):
        obj = getattr(c, "obj", None)
        if obj is None:
            detail[name] = "?"
            continue
        # Walk the MRO: several calculators (the PromptCalculator family) declare
        # only meta_info/__init__ on the subclass and inherit __call__ -- and the
        # model invocation lives in that inherited __call__.
        src = ""
        for klass in inspect.getmro(obj):
            if klass.__name__ in ("object", "StatCalculator", "ABC"):
                continue
            try:
                src += inspect.getsource(klass)
            except (OSError, TypeError):
                continue
        if not src:
            detail[name] = "?"
            continue
        n = len(re.findall(r"model\.generate\(", src)) + len(
            re.findall(r"model\.model\(|model\(\*\*|model\.__call__\(", src)
        )
        n = 1 if n > 1 else n  # multiple call sites are branches, not repeats
        detail[name] = n
        total += n
    return total, detail


def main():
    global ATTN_BATCH, ATTN_SEQ_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimators", type=Path, default=DEFAULT_ESTIMATORS)
    ap.add_argument("--out-dir", type=Path, default=REPO / "harness")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Batch size the feasibility verdict is evaluated at.")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="Prompt length; defaults to the MEASURED p99 in prompt_length.json.")
    ap.add_argument("--tag", default="", help="Suffix for the output filenames.")
    args = ap.parse_args()

    ATTN_BATCH = args.batch_size
    measured = REPO / "harness" / "prompt_length.json"
    if args.seq_len is not None:
        ATTN_SEQ_LEN = args.seq_len
        src = "--seq-len"
    elif measured.exists():
        import json as _json
        ATTN_SEQ_LEN = int(_json.loads(measured.read_text())["p99"])
        src = f"measured p99 ({measured.name})"
    else:
        src = "module default (UNMEASURED)"
    print(f"[tiers] batch_size={ATTN_BATCH}  prompt_len={ATTN_SEQ_LEN}  <- {src}")

    if not args.estimators.exists():
        sys.exit(f"estimator list not found: {args.estimators}")

    from lm_polygraph.utils.factory_estimator import FactoryEstimator

    entries = yaml.safe_load(args.estimators.read_text(encoding="utf-8")) or []
    print(f"[tiers] population: {len(entries)} rows from {args.estimators.name}")

    _, index = build_stat_index()
    factory = FactoryEstimator()

    rows = []
    for e in entries:
        name = e["name"]
        cfg = e.get("cfg") or {}
        cfg_str = ";".join(f"{k}={v}" for k, v in sorted(flatten_cfg(cfg))) or "-"

        # Classify from the declared cfg FIRST. Some estimators download a large
        # corpus in __init__ (Focus pulls RedPajama + a spaCy model), so
        # instantiating everything would trigger the very cost we are here to
        # rule out. If the cfg already marks it external, skip construction.
        _, precheck_flags, _, _ = classify(set(), {}, cfg)
        if "needs_external_corpus" in precheck_flags:
            deps, label, level = [], name, "?"
            err = "skipped: cfg declares an external corpus; not instantiated"
            print(f"[tiers] SKIP {name}: {err}")
        else:
            try:
                inst = factory(name, dict(cfg))
                deps = list(inst.stats_dependencies)
                label = str(inst)
                level = inst.level
                err = ""
            except Exception as exc:
                deps, label, level = [], name, "?"
                err = f"{type(exc).__name__}: {exc}"
                print(f"[tiers] WARN could not instantiate {name} ({cfg_str}): {err}")

        stats, calcs, unresolved = resolve_transitive(deps, index)
        tier, flags, attn_bytes, batch_peak = classify(stats, calcs, cfg)
        n_passes, _pass_detail = count_model_passes(calcs)
        # An instantiation failure must not erase a valid cfg-based verdict:
        # only fall back to the sentinel tier when nothing else classified it.
        if err and tier == "single_pass_cheap" and not flags:
            tier, flags = "instantiation_failed", {"instantiation_failed"}

        rows.append({
            "estimator": label,
            "name": name,
            "cfg": cfg_str,
            "level": level,
            "direct_deps": ",".join(sorted(deps)) or "-",
            "resolved_calculators": ",".join(sorted(calcs)) or "-",
            "n_resolved_stats": len(stats),
            "attn_mb_per_sample": round(attn_bytes / 1024**2, 1) if attn_bytes else 0,
            "attn_peak_mb_at_batch": round(batch_peak / 1024**2, 1) if batch_peak else 0,
            "model_passes": n_passes,
            "unresolved_stats": ",".join(sorted(unresolved)) or "-",
            "flags": ",".join(sorted(flags)) or "-",
            "tier": tier,
            "exclusion_reason": EXCLUSION_REASON.get(tier, err or ""),
            "error": err,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"estimator_tiers{args.tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[tiers] wrote {csv_path}")

    # markdown for the report
    md = ["# Baseline feasibility tiers (mechanically derived)", "",
          f"Population: all {len(rows)} rows of upstream "
          f"`examples/configs/estimators/{args.estimators.name}`. Each name+cfg pair is a "
          "distinct row. Tier is assigned by predicates over each estimator's *resolved "
          "transitive dependency set*, not by a list of names.", ""]

    order = TIER_PRIORITY + ["unsafe_attention_memory", "single_pass_cheap", "instantiation_failed"]
    PRIMARY = {"single_pass_cheap", "single_pass_plus_aux_model"}
    counts = {t: sum(1 for r in rows if r["tier"] == t) for t in order}
    md += ["| Tier | Count | Meaning |", "|---|---:|---|"]
    for t in order:
        if counts.get(t):
            md.append(f"| `{t}` | {counts[t]} | {EXCLUSION_REASON.get(t,'') or 'Primary baseline set.'} |")
    md.append("")

    for t in order:
        sub = [r for r in rows if r["tier"] == t]
        if not sub:
            continue
        md += [f"## {t}  ({len(sub)})", "",
               "| Estimator | cfg | Resolved calculators | Flags |", "|---|---|---|---|"]
        for r in sorted(sub, key=lambda r: r["estimator"]):
            md.append(
                f"| `{r['estimator']}` | {r['cfg']} | {r['resolved_calculators']} | {r['flags']} |"
            )
        md.append("")

    md_path = args.out_dir / f"estimator_tiers{args.tag}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[tiers] wrote {md_path}")

    print("\n=== tier counts ===")
    for t in order:
        if counts.get(t):
            print(f"  {t:24s} {counts[t]}")
    print("\n=== PRIMARY BASELINE SET (single_pass_cheap + single_pass_plus_aux_model) ===")
    for r in sorted(rows, key=lambda r: r["estimator"]):
        if r["tier"] in PRIMARY:
            extra = f"   [{r['flags']}]" if r["flags"] != "-" else ""
            print(f"  {r['estimator']:42s} passes={r['model_passes']}{extra}")


if __name__ == "__main__":
    main()
