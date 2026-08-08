import torch
import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class EnergyCalculator(StatCalculator):
    """
    Computes the energy statistics required by the Spilled Energy method
    (Minut, Dewidar & Masi, ICLR 2026) from the model's **raw logits**.

    Why this calculator has to exist
    --------------------------------
    ``GreedyProbsCalculator`` stores ``greedy_log_probs``, but those are
    ``log_softmax(logits)``: ``WhiteboxModel.generate`` installs a
    ``_ScoresProcessor`` that overwrites ``out.scores`` with
    ``scores.log_softmax(-1)``. Since ``log_softmax(theta) = theta - Z`` with
    ``Z = logsumexp_k theta[k]``, the log-partition ``Z`` is normalised away
    (``logsumexp`` of any log-prob row is exactly 0) and is **not recoverable**
    from any existing statistic. The Spilled Energy scores need ``Z``, so we
    re-run the model teacher-forced over ``prompt + greedy generation`` and read
    the raw logits before any normalisation.

    Statistics produced (per sample, generation of length ``N``)
    ------------------------------------------------------------
    * ``energy_token_logits`` : ``float32[N]`` -- raw logit of the *sampled*
      token at each generation step, ``theta_j[id(x_j)]``. The logit energy of
      the paper is ``E^l_j = -energy_token_logits[j]``.
    * ``energy_lse`` : ``float32[N + 1]`` -- log-partition
      ``Z_j = logsumexp_k theta_j[k]`` at each generation step, **plus one extra
      entry** for the step immediately after the last generated token. The
      marginal energy is ``E^m_j = -energy_lse[j]``. The trailing entry is what
      makes the *adjacent-step* spilled energy defined for the final token:
      ``dE_j = Z_{j+1} - theta_j[id(x_j)]``.

    Only these scalars are kept -- the ``[N, V]`` logit matrix is reduced on the
    fly and never stored (a 3B model with ``V ~ 152k`` would otherwise cost
    ~12 MB per sample).
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return ["energy_token_logits", "energy_lse"], ["greedy_tokens"]

    def __init__(self, batch_chunk_size: int = 0):
        """
        Parameters:
            batch_chunk_size (int): if > 0, run the teacher-forced pass in chunks
                of this many sequences to bound peak memory. 0 means one pass over
                the whole batch.
        """
        super().__init__()
        self.batch_chunk_size = batch_chunk_size

    def _prompt_ids(self, model: WhiteboxModel, texts: List[str]) -> List[List[int]]:
        """Tokenize each prompt individually (no padding) to get true lengths."""
        tokenizer = model.tokenizer
        add_special = getattr(tokenizer, "add_bos_token", None)
        ids = []
        for t in texts:
            enc = tokenizer(t, add_special_tokens=True, return_attention_mask=False)
            ids.append(list(enc["input_ids"]))
        del add_special
        return ids

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Runs a single teacher-forced forward pass over ``prompt + generation`` and
        reduces the raw logits to the two per-step scalar sequences described above.

        Parameters:
            dependencies (Dict[str, np.ndarray]): must contain 'greedy_tokens'.
            texts (List[str]): input texts batch used for the generation.
            model (WhiteboxModel): the model used for generation.
        Returns:
            Dict[str, np.ndarray]: 'energy_token_logits' and 'energy_lse'.
        """
        if getattr(model, "model_type", "CausalLM") != "CausalLM":
            raise NotImplementedError(
                "EnergyCalculator currently supports CausalLM models only; "
                f"got model_type={getattr(model, 'model_type', None)!r}."
            )

        greedy_tokens = dependencies["greedy_tokens"]
        prompt_ids = self._prompt_ids(model, texts)

        tokenizer = model.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        device = model.device()

        # Build right-padded teacher-forcing batch: [prompt || generation].
        # Right padding keeps every real position's causal context pad-free, so
        # per-sample indexing needs only that sample's own prompt length.
        seqs = [list(p) + list(g) for p, g in zip(prompt_ids, greedy_tokens)]
        lengths = [len(s) for s in seqs]
        max_len = max(lengths) if lengths else 0

        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, : len(s)] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        chunk = self.batch_chunk_size if self.batch_chunk_size > 0 else len(seqs)
        token_logits_out: List[np.ndarray] = []
        lse_out: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(seqs), chunk):
                stop = min(start + chunk, len(seqs))
                out = model.model(
                    input_ids=input_ids[start:stop],
                    attention_mask=attention_mask[start:stop],
                )
                logits = out.logits  # [b, max_len, V]

                for local_i in range(stop - start):
                    i = start + local_i
                    p_len = len(prompt_ids[i])
                    n_gen = len(greedy_tokens[i])
                    if n_gen == 0:
                        token_logits_out.append(np.zeros(0, dtype=np.float32))
                        lse_out.append(np.zeros(0, dtype=np.float32))
                        continue

                    # Position p_len - 1 + j predicts generated token j.
                    # Take j = 0..n_gen-1 for the sampled-token logits, and
                    # j = 0..n_gen for the log-partitions (one extra step).
                    first = p_len - 1
                    last = p_len - 1 + n_gen  # inclusive -> +1 in the slice
                    rows = logits[local_i, first : last + 1, :].float()  # [n_gen+1, V]

                    lse = torch.logsumexp(rows, dim=-1)  # [n_gen+1]

                    tok = torch.tensor(
                        list(greedy_tokens[i]), dtype=torch.long, device=rows.device
                    )
                    tok_logits = rows[:n_gen].gather(1, tok.unsqueeze(1)).squeeze(1)

                    token_logits_out.append(
                        tok_logits.cpu().numpy().astype(np.float32, copy=False)
                    )
                    lse_out.append(lse.cpu().numpy().astype(np.float32, copy=False))

                del out, logits

        return {
            "energy_token_logits": token_logits_out,
            "energy_lse": lse_out,
        }
