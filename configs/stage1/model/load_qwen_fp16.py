"""Stage 1 model loader for Qwen2.5-3B-Instruct on a 16 GB T4.

Binding constraints (see docs/stage1_recon.md and the Stage 1 brief):
  * fp16 only  -> torch_dtype=torch.float16  (NOT bfloat16, NOT fp32)
  * no FlashAttention-2 -> attn_implementation="eager"
  * single T4 -> device_map from config ("cuda" or "auto")

The stock lm-polygraph loaders (examples/configs/model/default_causal.py) do
NOT set torch_dtype, so a 3B model would materialise in fp32 (~24 GB) and OOM a
T4. This loader exists solely to pin fp16 + eager attention without editing any
upstream file. Keep it dependency-free and side-effect-free.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16,   # fp16, per T4 constraint
        attn_implementation="eager",  # no FlashAttention-2
    )
    model.eval()
    return model


def load_tokenizer(model_path: str, add_bos_token: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_bos_token=add_bos_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- align the EOS id with the token generation actually terminates on ----
    # Qwen2.5-*-Instruct declares two terminators:
    #     tokenizer.eos_token_id      = 151645  <|im_end|>     (chat turn end)
    #     generation_config.eos_token_id = [151645, 151643]
    #     tokenizer.pad_token_id      = 151643  <|endoftext|>
    #
    # We prompt in continuation style (instruct: false), so the model ends a
    # completion with <|endoftext|> (151643), never <|im_end|>. generate() then
    # pads the finished sequence out to max_new_tokens using pad_token_id, which
    # is also 151643.
    #
    # GreedyProbsCalculator trims the generation by scanning for a single id:
    #     if seq[j] == model.tokenizer.eos_token_id
    # With eos_token_id left at 151645 that scan never matches, so NOTHING is
    # trimmed. Observed consequences on the n=150 run:
    #   * at_ceiling_frac = 1.0 and gen_len p50 = p95 = max = 20 (looks as if
    #     stop_strings never fired; in fact it fired and the rest is padding)
    #   * "<|endoftext|>" left inside greedy_texts, so exact match can never
    #     succeed -- sufficient on its own for accuracy 0
    #   * ~18 padding tokens per sample fed into the energy statistics
    #
    # Pointing eos_token_id at the id that actually terminates continuation-style
    # generation makes the upstream trim work as intended. Upstream uses this same
    # extension point for Gemma (see examples/configs/model/gemma_3.py, which
    # reassigns eos_token_id to <end_of_turn>), so this is the sanctioned place
    # for it rather than a patch to the calculator.
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        tokenizer.eos_token_id = eot

    return tokenizer
