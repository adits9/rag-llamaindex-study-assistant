"""Small CLI entrypoint to run HF generation in an isolated process.

Reads JSON from stdin with fields: model_name, prompt, max_new_tokens, temperature
and writes JSON to stdout: {"text": "..."}.

This keeps the main Gradio process isolated from any native crashes inside
transformers/torch by running generation in a subprocess.
"""
from __future__ import annotations

import json
import sys
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    try:
        payload = json.load(sys.stdin)
    except Exception:
        print(json.dumps({"error": "invalid input"}))
        sys.exit(1)

    model_name = payload.get("model_name", "distilgpt2")
    prompt = payload.get("prompt", "")
    max_new_tokens = int(payload.get("max_new_tokens", 128))
    temperature = float(payload.get("temperature", 0.0))

    # limit threads
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # choose model class
    is_seq2seq = any(x in model_name.lower() for x in ("t5", "flan"))
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    try:
        if is_seq2seq:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
            text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        else:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=temperature>0.0)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(2)

    print(json.dumps({"text": text}))


if __name__ == "__main__":
    main()
