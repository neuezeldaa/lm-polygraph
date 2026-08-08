#!/usr/bin/env python3
"""Tiny-stub CPU smoke test for the full lm-polygraph pipeline.

Runs the ENTIRE UEManager pipeline -- stat calculators, dependency graph,
estimators (baselines + SpilledEnergy), generation metrics, PRR -- on a ~2.4M
parameter random-weight Qwen2 stub, on CPU, in seconds.

The numbers are meaningless by construction (random weights). The point is to
catch, without a Colab round trip:
  * StatCalculator registration and dependency-graph resolution,
  * stats key names and tensor/array shapes,
  * __call__ signature drift,
  * the structure of the saved manager (verifies the runner's parsing assumptions).

It writes a real ``ue_manager_seed*`` file so the offline analysis path
(``stage1_run_baselines.py --skip-run``) can be exercised locally.

Usage:
    python harness/smoke_tiny.py [--out DIR] [--n 8]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

QA = [
    ("Q: What is the capital of France?\nA:", "Paris"),
    ("Q: Who wrote Hamlet?\nA:", "Shakespeare"),
    ("Q: What is the largest planet?\nA:", "Jupiter"),
    ("Q: What year did WW2 end?\nA:", "1945"),
    ("Q: What is the chemical symbol for gold?\nA:", "Au"),
    ("Q: Who painted the Mona Lisa?\nA:", "Leonardo da Vinci"),
    ("Q: What is the tallest mountain?\nA:", "Everest"),
    ("Q: What is the smallest prime?\nA:", "2"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("workdir/smoke"))
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    import torch
    from lm_polygraph.utils.model import WhiteboxModel
    from lm_polygraph.utils.dataset import Dataset
    from lm_polygraph.utils.manager import UEManager
    from lm_polygraph.utils.builder_enviroment_stat_calculator import (
        BuilderEnvironmentStatCalculator,
    )
    from lm_polygraph.utils.factory_estimator import FactoryEstimator
    from lm_polygraph.defaults.register_default_stat_calculators import (
        register_default_stat_calculators,
    )
    from lm_polygraph.generation_metrics import AccuracyMetric
    from lm_polygraph.ue_metrics import PredictionRejectionArea
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[smoke] loading stub {TINY_MODEL} on CPU ...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        TINY_MODEL, attn_implementation="eager"
    )
    hf_model.eval()
    tok = AutoTokenizer.from_pretrained(TINY_MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from lm_polygraph.utils.generation_parameters import GenerationParameters

    model = WhiteboxModel(
        hf_model,
        tok,
        model_path=TINY_MODEL,
        model_type="CausalLM",
        generation_parameters=GenerationParameters(do_sample=False, num_beams=1),
    )

    xs = [q for q, _ in QA][: args.n]
    ys = [a for _, a in QA][: args.n]
    data = Dataset(xs, ys, batch_size=4)

    factory = FactoryEstimator()
    estimators = [
        factory("MaximumSequenceProbability", {}),
        factory("Perplexity", {}),
        factory("MeanTokenEntropy", {}),
        factory("SelfCertainty", {}),
    ]
    # the full ablation ladder: each adjacent rung differs by one ingredient
    for pooling in ("min", "max", "mean"):
        estimators.append(factory("harness.pooled_baseline",
                                  {"score": "log_likelihood", "pooling": pooling}))
    for variant in ("logit", "marginal", "spilled", "scaled_spilled"):
        for pooling in ("min", "max", "mean"):
            estimators.append(factory("SpilledEnergy",
                                      {"variant": variant, "pooling": pooling}))
    for pooling in ("min", "max", "mean"):
        estimators.append(factory("harness.pooled_baseline",
                                  {"score": "entropy", "pooling": pooling}))
    # terminator-exclusion variants must resolve the extra stat end-to-end
    estimators.append(factory("SpilledEnergy",
                              {"variant": "logit", "pooling": "min",
                               "exclude_terminator": True}))
    estimators.append(factory("harness.pooled_baseline",
                              {"score": "log_likelihood", "pooling": "min",
                               "exclude_terminator": True}))
    print(f"[smoke] estimators: {[str(e) for e in estimators]}")

    scs = register_default_stat_calculators(
        "Whitebox", output_attentions=False, output_hidden_states=False
    )
    names = [sc.name for sc in scs]
    assert "EnergyCalculator" in names, "EnergyCalculator not registered!"
    print("[smoke] EnergyCalculator IS registered")

    man = UEManager(
        data=data,
        model=model,
        estimators=estimators,
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(model=model),
        available_stat_calculators=scs,
        generation_metrics=[AccuracyMetric()],
        ue_metrics=[PredictionRejectionArea(), PredictionRejectionArea(max_rejection=0.5)],
        processors=[],
        ignore_exceptions=False,
        max_new_tokens=6,
        verbose=False,
        save_stats=["greedy_texts", "greedy_tokens", "energy_token_logits", "energy_lse", "energy_trailing_terminators"],
    )

    print("[smoke] running pipeline ...")
    man()

    args.out.mkdir(parents=True, exist_ok=True)
    save_to = args.out / "ue_manager_seed1"
    man.save(str(save_to))
    print(f"[smoke] saved manager -> {save_to}")

    # ---- verify the structural assumptions the runner relies on ----------
    print("\n[smoke] === structural verification ===")
    blob = torch.load(str(save_to), weights_only=False)
    print("top-level type:", type(blob).__name__)
    print("top-level keys:", list(blob.keys()) if isinstance(blob, dict) else "N/A")

    metrics = blob["metrics"] if isinstance(blob, dict) else blob.metrics
    ue_names = sorted({k[3] for k in metrics if isinstance(k, tuple) and len(k) == 4})
    print("distinct ue_metric names:", ue_names)
    example = next(iter(metrics))
    print("example key:", example, "-> type", type(example).__name__, "len", len(example))

    assert "prr_0.5_normalized" in ue_names, (
        f"'prr_0.5_normalized' NOT among {ue_names} -- runner parsing would fail!"
    )
    print("OK: 'prr_0.5_normalized' present, keys are 4-tuples")

    stats = blob["stats"] if isinstance(blob, dict) else blob.stats
    for k in ("energy_token_logits", "energy_lse", "greedy_tokens"):
        v = stats.get(k)
        print(f"stats[{k!r}]: n={len(v) if v is not None else None}", end="")
        if v:
            print(f" first_len={len(v[0])}")
        else:
            print()

    # the +1 invariant that makes adjacent-step dE well defined
    et, el, gt = stats["energy_token_logits"], stats["energy_lse"], stats["greedy_tokens"]
    for i in range(len(et)):
        assert len(et[i]) == len(gt[i]), f"sample {i}: token_logits {len(et[i])} != gen {len(gt[i])}"
        assert len(el[i]) == len(gt[i]) + 1, f"sample {i}: lse {len(el[i])} != gen+1 {len(gt[i])+1}"
    print("OK: len(energy_token_logits)==N and len(energy_lse)==N+1 for all samples")

    print("\n[smoke] === PRR@0.5 (MEANINGLESS -- random weights) ===")
    for k, v in sorted(metrics.items()):
        if len(k) == 4 and k[3] == "prr_0.5_normalized":
            print(f"  {k[1]:40s} {v: .4f}")

    print("\n[smoke] PASS")


if __name__ == "__main__":
    sys.exit(main())
