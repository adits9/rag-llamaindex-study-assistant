"""Embedding adapter that wraps sentence-transformers for use with LlamaIndex.

Implements a small subclass of LlamaIndex BaseEmbedding so we can pass this
instance to index construction and retrieval. Forces CPU usage and limits
threads to avoid macOS MPS / multiprocessing segfaults.
"""
from __future__ import annotations

import asyncio
import os
from typing import List

# LlamaIndex base embedding
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - runtime guard
    raise ImportError("Please install sentence-transformers to use embeddings: pip install sentence-transformers") from exc


# Small safety defaults for macOS: force CPU and limit threads.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")


class SentenceTransformersEmbedding(BaseEmbedding):
    """Wraps sentence-transformers as a LlamaIndex BaseEmbedding.

    Example:
        emb = SentenceTransformersEmbedding(model_name="all-MiniLM-L6-v2")
        index = VectorStoreIndex.from_documents(docs, embed_model=emb, ...)
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None, **kwargs) -> None:
        # pydantic-backed BaseEmbedding expects fields passed into super().__init__
        model_name_val = model_name or self.model_name
        # set small batch size to be conservative on CPU
        super().__init__(model_name=model_name_val, embed_batch_size=64, **kwargs)

        # load model on CPU explicitly
        # self.model_name is now available from the pydantic model
        self._model = SentenceTransformer(self.model_name, device="cpu")

    def _get_text_embedding(self, text: str) -> Embedding:
        # returns list[float]
        vec = self._model.encode(text, show_progress_bar=False)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        # run blocking encode in thread to avoid blocking event loop
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await self._aget_text_embedding(query)
