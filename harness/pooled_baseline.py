"""PooledBaseline -- the ablation control for Spilled Energy.

SpilledEnergy benefits from two things at once: the energy formulation, and
localisation to the answer window with min/max/mean pooling. Comparing it against
library baselines that sum or average over the whole sequence confounds the two --
a win cannot be attributed to either ingredient.

This applies the SAME three poolings over the SAME window to lm-polygraph's own
token-level scores, so the only remaining difference is the energy formulation.

Deliberately kept OUT of the PR branch. lm-polygraph has no pooling abstraction --
every estimator hardcodes its own sum/mean -- so a general pooling API is a real
gap, but proposing one unsolicited invites an API design debate that would stall
the method PR. This is evaluation scaffolding; if the maintainers want a pooling
primitive it deserves its own PR with their input.

Loaded by dotted path, so no upstream registration is needed:

    - name: harness.pooled_baseline
      cfg: {score: log_likelihood, pooling: max}

Window: the whole generation. With ``stop_strings: ["\\n"]`` and
``max_new_tokens: 20`` the TriviaQA generation IS the short answer, which mirrors
how the paper obtains its [u, w] span (by prompting for a brief answer) rather
than approximating it. Crucially this uses NO ground truth -- a span located by
matching the gold answer would be label leakage and unavailable at inference.
Validate the assumption with harness/validate_answer_span.py before reporting.
"""

import numpy as np

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator

SCORES = ("log_likelihood", "entropy")
POOLINGS = ("min", "max", "mean")

# stat each score reads, and whether higher = more uncertain before the sign flip
_SCORE_STATS = {
    "log_likelihood": "greedy_log_likelihoods",
    "entropy": "entropy",
}


class PooledBaseline(Estimator):
    """A token-level lm-polygraph score, pooled over the generation window.

    Parameters:
        score: 'log_likelihood' (the MSP/Perplexity family) or 'entropy'
            (the MeanTokenEntropy family).
        pooling: 'min' | 'max' | 'mean', matching SpilledEnergy's options.
        sign: +1 or -1. The Estimator contract is higher = more uncertain.
            log-likelihood is negated by default so that low probability reads as
            high uncertainty (matching MaximumSequenceProbability); entropy is
            already oriented that way.
    """

    def __init__(self, score: str = "log_likelihood", pooling: str = "max",
                 sign: int = None, exclude_terminator: bool = False):
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {score!r}")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")

        deps = [_SCORE_STATS[score]]
        if exclude_terminator:
            # produced by EnergyCalculator, which knows the tokenizer
            deps.append("energy_trailing_terminators")
        super().__init__(deps, "sequence")
        self.score = score
        self.pooling = pooling
        self.exclude_terminator = exclude_terminator
        if sign is None:
            sign = -1 if score == "log_likelihood" else 1
        if sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {sign!r}")
        self.sign = sign

    def __str__(self):
        tag = "" if self.sign == (-1 if self.score == "log_likelihood" else 1) else "_flip"
        term = "_noterm" if self.exclude_terminator else ""
        return f"Pooled_{self.score}_{self.pooling}{tag}{term}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        per_token = stats[_SCORE_STATS[self.score]]
        trailing = (
            stats["energy_trailing_terminators"] if self.exclude_terminator
            else [0] * len(per_token)
        )
        pool = {"min": np.min, "max": np.max, "mean": np.mean}[self.pooling]

        out = []
        for seq, n_term in zip(per_token, trailing):
            arr = np.asarray(seq, dtype=np.float64)
            # Same window as SpilledEnergy: the ablation is only valid if BOTH
            # sides drop the terminator, otherwise the comparison changes two
            # things at once.
            if n_term:
                arr = arr[: max(1, len(arr) - int(n_term))]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                out.append(np.nan)
                continue
            # Orient to uncertainty BEFORE pooling, exactly as SpilledEnergy does
            # (its _per_token_scores returns -theta[id], -Z, ... and pools those).
            # Pooling the raw score and negating afterwards would make "max" mean
            # the opposite thing in the two estimators and break the ablation.
            out.append(float(pool(self.sign * arr)))
        return np.array(out, dtype=np.float64)


def load_estimator(config):
    """Entry point used by FactoryEstimator when loading by dotted module path."""
    cfg = dict(config) if config else {}
    return PooledBaseline(**cfg)
