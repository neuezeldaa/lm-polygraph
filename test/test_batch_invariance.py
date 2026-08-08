"""Batching must not change a generation.

Running a prompt in a batch alongside a longer prompt must produce exactly what
running it alone produces. That property is what a padding-side or attention-mask
error breaks, and nothing else in the suite checks it.

Motivation: on the first T4 run, 108/150 samples generated only token id 0 ('!'),
and the failures were exactly the shorter prompts in each batch (42/150 good =
28%, against 25% expected if only the longest of each 4-sample batch survives).

Runs on CPU against the tiny random-weight Qwen2 stub, so it is also a
discriminating experiment: this stub runs in float32, where no fp16 range
overflow is possible. If batching changes the output HERE, the cause is
structural (padding side / mask). If it does not, the cause is numerical and
specific to fp16.
"""

import numpy as np
import pytest
import torch

TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

SHORT = "Question: What is the capital of France?\nAnswer:"
LONG = (
    "Question: Search for Extra-Terrestrial Intelligence (SETI) is the collective "
    "name for a number of activities to search for extra-terrestrial life using "
    "scientific methods to search for what, specifically?\nAnswer:"
)


@pytest.fixture(scope="module")
def model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_polygraph.utils.model import WhiteboxModel
    from lm_polygraph.utils.generation_parameters import GenerationParameters

    try:
        hf = AutoModelForCausalLM.from_pretrained(TINY_MODEL, attn_implementation="eager")
        tok = AutoTokenizer.from_pretrained(TINY_MODEL, padding_side="left")
    except Exception as e:
        pytest.skip(f"tiny stub unavailable: {e}")
    hf.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return WhiteboxModel(
        hf, tok, model_path=TINY_MODEL, model_type="CausalLM",
        generation_parameters=GenerationParameters(do_sample=False, num_beams=1),
    )


def test_padding_side_is_left(model):
    """Decoder-only generation requires left padding.

    With right padding a short sequence ends in pad tokens, so generation
    continues from a position whose immediate context is padding.
    """
    assert model.tokenizer.padding_side == "left", (
        f"padding_side is {model.tokenizer.padding_side!r}; decoder-only "
        "generation must pad on the left"
    )


def test_attention_mask_marks_exactly_the_padding(model):
    """The mask must be 0 on pad positions and 1 everywhere else, on the correct side."""
    batch = model.tokenize([SHORT, LONG])
    ids, mask = batch["input_ids"], batch["attention_mask"]
    pad_id = model.tokenizer.pad_token_id

    assert ids.shape == mask.shape
    for row_ids, row_mask in zip(ids, mask):
        # every masked-out position must actually be a pad token
        assert (row_ids[row_mask == 0] == pad_id).all(), "mask hides a non-pad token"
        # padding must be a prefix (left side), never a suffix
        zeros = (row_mask == 0).nonzero().flatten().tolist()
        if zeros:
            assert zeros == list(range(len(zeros))), (
                f"padding is not a left-aligned prefix: masked positions {zeros[:8]}"
            )
        # the final position must always be real content
        assert row_mask[-1] == 1


def test_batched_generation_matches_individual(model):
    """The short prompt must generate the same tokens batched as it does alone."""
    from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator

    calc = GreedyProbsCalculator(output_attentions=False, output_hidden_states=False)

    batched = calc({}, [SHORT, LONG], model, max_new_tokens=6)
    alone_short = calc({}, [SHORT], model, max_new_tokens=6)
    alone_long = calc({}, [LONG], model, max_new_tokens=6)

    assert batched["greedy_tokens"][0] == alone_short["greedy_tokens"][0], (
        "the SHORT prompt generated different tokens when batched with a longer "
        "one -- padding is corrupting its context"
    )
    assert batched["greedy_tokens"][1] == alone_long["greedy_tokens"][0], (
        "the LONG prompt generated different tokens when batched"
    )

    # log-likelihoods must agree closely too (fp tolerance, not bit equality)
    np.testing.assert_allclose(
        np.asarray(batched["greedy_log_likelihoods"][0], dtype=np.float64),
        np.asarray(alone_short["greedy_log_likelihoods"][0], dtype=np.float64),
        atol=1e-4, rtol=0,
        err_msg="batched log-likelihoods diverge from unbatched for the short prompt",
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fully_masked_attention_row_does_not_overflow(dtype):
    """A fully-masked attention row must not become NaN in the run dtype.

    This is the actual root cause of the 108/150 collapse, and it is checkable
    without a GPU. Qwen2 eager attention computes

        attn_weights = attn_weights + causal_mask,   causal_mask = finfo(dtype).min

    Left padding makes the leading query positions attend to nothing, so their
    whole row is masked. In fp16, finfo.min is -65504 and adding any ordinary
    negative score overflows the row to -inf; softmax over an all -inf row is
    NaN. In fp32 the same arithmetic stays finite.

    transformers guards exactly this via AttentionMaskConverter._unmask_unattended,
    but only when _attn_implementation == "sdpa" AND output_attentions is False.
    The Stage 1 config uses eager (per the no-FlashAttention-2 constraint) and
    needs output_attentions for the attention-based baselines, so neither
    condition holds and the guard never runs.
    """
    scores = torch.full((1, 8), -30.0, dtype=dtype)
    mask = torch.full((1, 8), torch.finfo(dtype).min, dtype=dtype)
    summed = scores + mask
    probs = torch.softmax(summed, dim=-1)

    if dtype == torch.float16:
        # documents the failure rather than asserting it away
        assert torch.isinf(summed).any(), "expected fp16 overflow to -inf"
        assert torch.isnan(probs).any(), "expected NaN from an all -inf softmax"
    else:
        assert torch.isfinite(summed).all()
        assert not torch.isnan(probs).any()


def test_sanitizer_turns_non_finite_logits_into_a_uniform_distribution():
    """Pin the mechanism that HID the NaN, so it cannot mislead a future diagnosis.

    lm-polygraph's _SanitizeLogitsProcessor replaces a fully non-finite score row
    with zeros, which log_softmax turns into -ln(V) at every position. That is why
    the corrupted samples reported finite log-likelihoods, identical to the last
    digit, and why a finiteness check on greedy_log_probs can never detect the
    problem: by the time those values are recorded, the NaN has been erased.
    """
    from lm_polygraph.utils.model import WhiteboxModel

    san = WhiteboxModel._SanitizeLogitsProcessor()
    V = 151936
    for bad in (float("nan"), float("inf"), float("-inf")):
        out = san(None, torch.full((1, V), bad))
        assert torch.isfinite(out).all()
        assert (out == out[0, 0]).all(), "sanitised row is not uniform"
        lsm = out.log_softmax(-1)
        assert abs(lsm[0, 0].item() - (-np.log(V))) < 1e-5
        assert int(out.argmax()) == 0, "argmax of a uniform row is token 0 ('!')"


def test_generation_is_not_a_degenerate_constant(model):
    """Catch the observed failure shape directly: a flat distribution.

    When every logit is equal, log_softmax is -ln(V) at every position and argmax
    is token 0. lm-polygraph's _SanitizeLogitsProcessor produces exactly that from
    an all-non-finite row, which is why the corrupted samples had a log-likelihood
    of -ln(151936) identical to the last digit.
    """
    from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator

    calc = GreedyProbsCalculator(output_attentions=False, output_hidden_states=False)
    stats = calc({}, [SHORT, LONG], model, max_new_tokens=6)

    V = len(model.tokenizer)
    flat = -np.log(V)
    for i, lls in enumerate(stats["greedy_log_likelihoods"]):
        arr = np.asarray(lls, dtype=np.float64)
        assert np.isfinite(arr).all(), f"sample {i}: non-finite log-likelihood"
        assert not np.allclose(arr, flat, atol=1e-3), (
            f"sample {i}: log-likelihood is -ln(V) at every position -- the logit "
            "distribution is uniform, i.e. the scores were sanitised from "
            "non-finite values"
        )
        # a constant value across all steps is the same symptom, vocab-size aside
        if len(arr) > 2:
            assert arr.std() > 0, f"sample {i}: identical log-likelihood at every step"


def test_subsample_is_prefix_stable():
    """A smaller subsample must be the prefix of a larger one under the same seed.

    This is what makes the batch_size=1 attention run (n=300) comparable to the
    batch_size=4 primary run (n=1000): they see the same first 300 samples, so
    harness/check_run_consistency.py --allow-prefix can hash-compare them.
    Dataset.subsample uses np.random.choice(N, size, replace=False) after
    np.random.seed(seed); this pins the prefix property that relies on.
    """
    for N, seed in [(10000, 1), (2000, 1), (5000, 7)]:
        np.random.seed(seed)
        small = np.random.choice(N, 300, replace=False)
        np.random.seed(seed)
        large = np.random.choice(N, 1000, replace=False)
        assert np.array_equal(small, large[:300]), (
            f"subsample is not prefix-stable for N={N}, seed={seed}; the two runs "
            "would not share samples and could not be compared"
        )
