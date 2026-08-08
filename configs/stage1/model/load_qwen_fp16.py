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
    return tokenizer
