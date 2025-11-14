"""Lightweight generator wrapper using Hugging Face transformers.

This module contains HFGenerator, a tiny class that loads a local HF model on CPU
and exposes a .generate(prompt, max_new_tokens, temperature) method returning text.
It supports both causal and encoder-decoder (seq2seq) models by trying the
appropriate AutoModel classes.
"""
from __future__ import annotations

import os
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    logging as hf_logging,
)
import torch


hf_logging.set_verbosity_error()


import json
import subprocess
from typing import Optional


class HFGenerator:
    """Wraps a HF model for local generation on CPU.

    Args:
        model_name: HF model id (local or remote cache). Examples: 'distilgpt2',
            'google/flan-t5-base'.
        device: only 'cpu' supported in this project.
    """

    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = torch.device(device)

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # decide model class: prefer seq2seq if name contains 'flan' or 't5' else causal
        try:
            if any(x in model_name.lower() for x in ("t5", "flan")):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.is_seq2seq = True
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.is_seq2seq = False
        except Exception:
            # fallback to causal
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.is_seq2seq = False

        # run on CPU; use eval mode
        # Limit PyTorch threading to reduce chance of low-level crashes on macOS
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        self.model.to(self.device)
        self.model.eval()

        # keep generation defaults
        self.default_max_new_tokens = 256

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> str:
        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        # Run generation in a subprocess to isolate native crashes.
        payload = {
            "model_name": self.model_name,
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }

        proc = subprocess.run(
            [os.sys.executable, "-m", "rag_study_buddy.generator_cli"],
            input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )

        if proc.returncode != 0:
            # try to surface useful message
            try:
                err = proc.stdout.decode("utf-8") or proc.stderr.decode("utf-8")
            except Exception:
                err = proc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"Generator subprocess failed: {err}")

        out = proc.stdout.decode("utf-8")
        try:
            j = json.loads(out)
            if "text" in j:
                return j["text"]
            elif "error" in j:
                raise RuntimeError(f"Generator error: {j['error']}")
            else:
                raise RuntimeError(f"Unexpected generator output: {out}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid generator output: {out}")
