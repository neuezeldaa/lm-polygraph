"""TriviaQA exact-match normalization for Stage 1.

Vendored verbatim from lm-polygraph's
examples/configs/instruct/output_processing_scripts/triviaqa.py so that the
Stage 1 config directory is self-contained and portable to a fresh Colab clone
(paths in the Hydra config resolve relative to this directory). Only
`normalize_em_triviaqa` is used here; it is applied to both the model output and
the gold target before AccuracyMetric does an exact string match — i.e. standard
normalized exact match (lowercase, strip articles/punctuation/underscores,
collapse whitespace).
"""

import re
import string


def normalize_em_triviaqa(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(s))))
    ).strip()
