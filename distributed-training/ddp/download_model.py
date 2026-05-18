"""
Download a HuggingFace causal LM + tokenizer and print diagnostic information.

Usage:
    python download_model.py
    python download_model.py --model_id foo/bar
    python download_model.py --output_dir /path/to/save
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

## Config -----------------------------
MODEL_ID   = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "./models/Llama-3.2-1B-Instruct"

SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Explain gradient descent in one sentence.",
]


## Helpers -----------------------------
def human_size(num_params: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(num_params) < 1_000:
            return f"{num_params:.1f}{unit}"
        num_params /= 1_000
    return f"{num_params:.1f}T"


## Main -----------------------------
def main(model_id: str, output_dir: str):
    ## Tokenizer -----------------------------
    print(f"Downloading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  Class         : {type(tokenizer).__name__}")
    print(f"  Vocab size    : {tokenizer.vocab_size:,}")
    print(f"  Model max len : {tokenizer.model_max_length}")
    print(f"  BOS / EOS     : {tokenizer.bos_token!r} / {tokenizer.eos_token!r}")
    print(f"  PAD           : {tokenizer.pad_token!r}")
    print(f"  Chat template : {'yes' if tokenizer.chat_template else 'no'}")

    # Quick encode / decode round-trip
    sample = "Hello, distributed training!"
    ids = tokenizer.encode(sample)
    print(f"  Encode {sample!r}")
    print(f"    token ids : {ids}")
    print(f"    decoded   : {tokenizer.decode(ids)!r}")

    ## Model -----------------------------
    print(f"\nDownloading model: {model_id}  (bfloat16)")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.eval()

    ## Architecture -----------------------------
    cfg = model.config
    print(f"  Model type        : {cfg.model_type}")
    print(f"  Hidden size       : {cfg.hidden_size}")
    print(f"  Num hidden layers : {cfg.num_hidden_layers}")
    print(f"  Num attn heads    : {cfg.num_attention_heads}")
    kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    print(f"  KV heads (GQA)    : {kv}")
    print(f"  Intermediate size : {cfg.intermediate_size}")
    print(f"  Max position embs : {cfg.max_position_embeddings}")
    print(f"  Vocab size        : {cfg.vocab_size:,}")
    print(f"  Dtype             : {cfg.torch_dtype}")

    ## Parameters -----------------------------
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total      : {total:>15,}  ({human_size(total)})")
    print(f"  Trainable  : {trainable:>15,}  ({human_size(trainable)})")
    print(f"  Memory bf16: {total * 2 / 1e9:.2f} GB")
    print(f"  Memory fp32: {total * 4 / 1e9:.2f} GB")
    # AdamW stores fp32 master weights + 2 moment buffers ≈ 3× the fp32 model
    print(f"  + AdamW    : ~{total * 4 * 3 / 1e9:.2f} GB  (fp32 weights + 2 moments)")

    ## Layer breakdown -----------------------------
    print()
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<30} {params:>12,}  ({human_size(params)})")

    ## Sample inference -----------------------------
    print("\n## Sample inference (CPU, greedy, 64 tokens) -----------------------------")
    for prompt in SAMPLE_PROMPTS:
        messages  = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with torch.no_grad():
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][input_ids["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        print(f"\n  Prompt   : {prompt}")
        print(f"  Response : {generated.strip()}")

    ## Save -----------------------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel + tokenizer saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and inspect a causal LM")
    parser.add_argument("--model_id",   default=MODEL_ID,   help="HuggingFace model ID")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Where to save the model")
    args = parser.parse_args()

    main(args.model_id, args.output_dir)
