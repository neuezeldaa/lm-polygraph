"""Regression tests for generation trimming and decoding.

These pin the two failures that made the first T4 run produce accuracy 0, both
traceable to one mismatch: lm-polygraph trims a generation by scanning for
``tokenizer.eos_token_id`` (a single id), while ``generate()`` terminates on any
id in ``generation_config.eos_token_id`` and then pads with ``pad_token_id``.

When those disagree, nothing is trimmed and three things break at once:
  * every sequence looks as if it ran to max_new_tokens (at_ceiling_frac = 1.0)
  * special tokens survive in greedy_texts, so exact match can never succeed
  * padding tokens are fed into the energy statistics

Runs on CPU in milliseconds; only the tokenizer is needed, never model weights.
"""

import pytest

MODEL = "Qwen/Qwen2.5-3B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # offline CI
        pytest.skip(f"tokenizer unavailable: {e}")


def _trim(seq, eos_id):
    """Verbatim copy of the trim in GreedyProbsCalculator.__call__."""
    length = text_length = len(seq)
    for j in range(len(seq)):
        if seq[j] == eos_id:
            length = j + 1
            text_length = j
            break
    return length, text_length


def _load_our_tokenizer():
    """The tokenizer as the Stage 1 config actually builds it."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent
        / "configs" / "stage1" / "model" / "load_qwen_fp16.py"
    )
    spec = importlib.util.spec_from_file_location("load_qwen_fp16", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_tokenizer(MODEL, add_bos_token=False)


def test_eos_id_matches_the_padding_generate_actually_emits(tokenizer):
    """The trim id must equal the id generate() pads finished sequences with.

    This is the invariant whose violation caused the failure. The stock
    Instruct tokenizer breaks it: eos_token_id is <|im_end|> while pad_token_id
    (and the continuation-style terminator) is <|endoftext|>.
    """
    ours = _load_our_tokenizer()
    assert ours.eos_token_id == tokenizer.pad_token_id, (
        f"trim id {ours.eos_token_id} != pad id {tokenizer.pad_token_id}; "
        "padding will not be trimmed"
    )
    assert ours.eos_token_id == tokenizer.convert_tokens_to_ids("<|endoftext|>")


def test_padding_is_trimmed_so_generations_are_not_all_at_the_ceiling(tokenizer):
    """at_ceiling_frac must not be 1.0 for a short answer that finished early."""
    ours = _load_our_tokenizer()
    max_new = 20
    answer = tokenizer("iron\n", add_special_tokens=False)["input_ids"]
    seq = answer + [tokenizer.pad_token_id] * (max_new - len(answer))

    stock_len, _ = _trim(seq, tokenizer.eos_token_id)
    ours_len, _ = _trim(seq, ours.eos_token_id)

    assert stock_len == max_new, "precondition: the stock id fails to trim"
    assert ours_len < max_new, "padding still not trimmed"
    assert ours_len == len(answer) + 1  # answer plus the terminator itself


def test_no_special_tokens_survive_in_decoded_text(tokenizer):
    """greedy_texts must contain no special tokens; one is enough to break EM."""
    ours = _load_our_tokenizer()
    max_new = 20

    for answer_text in ["iron\n", "paris\n", "1945\n"]:
        answer = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        seq = answer + [tokenizer.pad_token_id] * (max_new - len(answer))

        _, text_len = _trim(seq, ours.eos_token_id)
        decoded = tokenizer.decode(seq[:text_len])

        for special in ("<|endoftext|>", "<|im_end|>", "<|im_start|>"):
            assert special not in decoded, f"{special!r} survived in {decoded!r}"
        assert decoded == answer_text


def test_exact_match_now_succeeds_end_to_end(tokenizer):
    """The decoded answer must survive TriviaQA normalisation to match the gold."""
    import importlib.util
    from pathlib import Path

    norm_path = (
        Path(__file__).resolve().parent.parent
        / "configs" / "stage1" / "triviaqa_normalization.py"
    )
    spec = importlib.util.spec_from_file_location("norm", norm_path)
    norm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(norm_mod)
    normalize = norm_mod.normalize_em_triviaqa

    ours = _load_our_tokenizer()
    answer = tokenizer("iron\n", add_special_tokens=False)["input_ids"]
    seq = answer + [tokenizer.pad_token_id] * (20 - len(answer))

    _, text_len = _trim(seq, ours.eos_token_id)
    assert normalize(tokenizer.decode(seq[:text_len])) == normalize("Iron")

    # and confirm the stock id would NOT have matched -- i.e. this test has teeth
    _, stock_text_len = _trim(seq, tokenizer.eos_token_id)
    assert normalize(tokenizer.decode(seq[:stock_text_len])) != normalize("Iron")
