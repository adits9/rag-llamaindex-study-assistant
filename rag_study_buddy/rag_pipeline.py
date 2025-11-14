"""RAG pipeline: loads persisted LlamaIndex index and answers queries.

The pipeline uses LlamaIndex for retrieval and a local HF generator for
generation. It exposes a simple `answer(question)` method that returns
the generated text and a list of source snippets.
"""
from __future__ import annotations

import os
from typing import List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.loading import load_index_from_storage

from .models.generator_model import HFGenerator


class RAGPipeline:
    def __init__(
        self,
        persist_dir: str = "./data/index",
        gen_model: str = "distilgpt2",
    ) -> None:
        self.persist_dir = persist_dir
        self.gen_model_name = gen_model
        self.generator = HFGenerator(self.gen_model_name, device="cpu")
        self._load_index()

    def _load_index(self) -> None:
        storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
        # Use our local sentence-transformers embedding when loading the index
        from .models.embedding_model import SentenceTransformersEmbedding

        embed = SentenceTransformersEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # load index (assumes single index); pass embed_model so llama-index doesn't try to use OpenAI
        self.index = load_index_from_storage(storage_context, embed_model=embed)
        self.retriever = self.index.as_retriever()

    def reload_index(self) -> None:
        """Reload the index from disk into memory. Call after a rebuild finished."""
        self._load_index()

    def answer(self, question: str, top_k: int = 4) -> tuple[str, List[str]]:
        """Return (answer_text, sources_list).

        For simplicity we ask the retriever for matching nodes and concatenate
        their text as context for the generator.
        """
        nodes_with_score = self.retriever.retrieve(question)
        top = nodes_with_score[:top_k]
        contexts = [n.node.get_content(metadata_mode=None) for n in top]
        sources = []
        for n in top:
            md = getattr(n.node, "metadata", None) or {}
            ref = md.get("source", None) or md.get("file_path", None) or n.node.ref_doc_id
            sources.append(str(ref))

        # build a simple prompt that includes context and the question
        context_block = "\n\n---\n\n".join(contexts)
        prompt = (
            "Use the following extracted context to answer the question. Be concise.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        )

        try:
            answer = self.generator.generate(prompt, max_new_tokens=256, temperature=0.0)
        except Exception as e:
            # If local generation fails (native crash in torch/transformers),
            # fall back to returning the context as a plain answer so the UI
            # remains responsive.
            answer = (
                "[Generation failed — showing supporting context instead]\n\n"
                + context_block
                + f"\n\n(Generation error: {e})"
            )

        return answer, sources
