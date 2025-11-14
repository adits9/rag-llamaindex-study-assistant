"""Document ingestion using LlamaIndex and sentence-transformers embeddings.

This script reads files from --input_dir (defaults to ./data/raw_docs),
builds a VectorStoreIndex using our SentenceTransformersEmbedding, and
persists the storage in --persist_dir (defaults to ./data/index).

Designed to be called as a background rebuild process by the UI.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# keep runtime safe for macOS: force CPU and reduce thread parallelism
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from llama_index.core import SimpleDirectoryReader
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.vector_store.base import GPTVectorStoreIndex

from .models.embedding_model import SentenceTransformersEmbedding


def build_index(
    input_dir: str = "./data/raw_docs",
    persist_dir: str = "./data/index",
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    print(f"Ingest: input_dir={input_dir} persist_dir={persist_dir} embed_model={embed_model_name}")

    reader = SimpleDirectoryReader(input_dir)
    documents = reader.load_data()

    # Create a fresh in-memory StorageContext, then persist to persist_dir
    # (calling from_defaults with persist_dir expects existing files).
    storage_context = StorageContext.from_defaults()

    embed = SentenceTransformersEmbedding(model_name=embed_model_name)

    # build VectorStoreIndex (GPTVectorStoreIndex is an alias)
    index = GPTVectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed
    )

    # persist storage context to disk (creates the directory/files)
    storage_context.persist(persist_dir)
    print(f"Built and persisted index with {len(documents)} documents to {persist_dir}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser("rag-ingest")
    parser.add_argument("--input_dir", default="./data/raw_docs")
    parser.add_argument("--persist_dir", default="./data/index")
    parser.add_argument(
        "--embed_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model name",
    )

    args = parser.parse_args(argv)
    build_index(input_dir=args.input_dir, persist_dir=args.persist_dir, embed_model_name=args.embed_model)


if __name__ == "__main__":
    main()
