"""Ingest documents into a FAISS-backed vector store using sentence-transformers.

This script reads PDFs and .txt files from ./data/raw_docs, chunks them, embeds
with a SentenceTransformer model, builds a FAISS index, and saves metadata.

Run: python ingest.py --rebuild
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


RAW_DIR = Path("./data/raw_docs")
STORAGE_DIR = Path("./storage")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50


def read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(p: Path) -> str:
    text = []
    reader = PdfReader(str(p))
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text)


def load_documents() -> Dict[str, str]:
    docs = {}
    if not RAW_DIR.exists():
        raise RuntimeError(f"Raw docs dir not found: {RAW_DIR}")
    for p in RAW_DIR.iterdir():
        if p.is_file():
            if p.suffix.lower() == ".txt":
                docs[str(p.name)] = read_text_file(p)
            elif p.suffix.lower() == ".pdf":
                docs[str(p.name)] = read_pdf_file(p)
            else:
                # ignore other file types for now
                continue
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    if length == 0:
        return []
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def build_index(embed_model_name: str = EMBED_MODEL_NAME):
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No documents found in {RAW_DIR}; add .txt or .pdf files.")

    model = SentenceTransformer(embed_model_name)

    texts = []
    metadatas = []
    for fname, content in docs.items():
        for i, chunk in enumerate(chunk_text(content)):
            texts.append(chunk)
            metadatas.append({"source": fname, "chunk": i})

    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(STORAGE_DIR / "index.faiss"))

    # save metadata and texts
    (STORAGE_DIR / "metadatas.json").write_text(json.dumps(metadatas, ensure_ascii=False))
    (STORAGE_DIR / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))

    # save config
    (STORAGE_DIR / "config.json").write_text(json.dumps({"embed_model": embed_model_name}))
    print(f"Built index with {len(texts)} chunks and saved to {STORAGE_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index")
    args = parser.parse_args()

    if args.rebuild or not (STORAGE_DIR.exists() and (STORAGE_DIR / "index.faiss").exists()):
        build_index()
    else:
        print("Index already exists in storage/. Use --rebuild to rebuild.")


if __name__ == "__main__":
    main()
"""Ingest documents into a FAISS-backed vector store using sentence-transformers.
"""Ingest documents into a FAISS-backed vector store using sentence-transformers.

This script reads PDFs and .txt files from ./data/raw_docs, chunks them, embeds
with a SentenceTransformer model, builds a FAISS index, and saves metadata.

Run: python ingest.py --rebuild
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


RAW_DIR = Path("./data/raw_docs")
STORAGE_DIR = Path("./storage")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50


def read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(p: Path) -> str:
    text = []
    reader = PdfReader(str(p))
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text)


def load_documents() -> Dict[str, str]:
    docs = {}
    if not RAW_DIR.exists():
        raise RuntimeError(f"Raw docs dir not found: {RAW_DIR}")
    for p in RAW_DIR.iterdir():
        if p.is_file():
            if p.suffix.lower() == ".txt":
                docs[str(p.name)] = read_text_file(p)
            elif p.suffix.lower() == ".pdf":
                docs[str(p.name)] = read_pdf_file(p)
            else:
                # ignore other file types for now
                continue
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    if length == 0:
        return []
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def build_index(embed_model_name: str = EMBED_MODEL_NAME):
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No documents found in {RAW_DIR}; add .txt or .pdf files.")

    model = SentenceTransformer(embed_model_name)

    texts = []
    metadatas = []
    for fname, content in docs.items():
        for i, chunk in enumerate(chunk_text(content)):
            texts.append(chunk)
            metadatas.append({"source": fname, "chunk": i})

    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(STORAGE_DIR / "index.faiss"))

    # save metadata and texts
    (STORAGE_DIR / "metadatas.json").write_text(json.dumps(metadatas, ensure_ascii=False))
    (STORAGE_DIR / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))

    # save config
    (STORAGE_DIR / "config.json").write_text(json.dumps({"embed_model": embed_model_name}))
    print(f"Built index with {len(texts)} chunks and saved to {STORAGE_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index")
    args = parser.parse_args()

    if args.rebuild or not (STORAGE_DIR.exists() and (STORAGE_DIR / "index.faiss").exists()):
        build_index()
    else:
        print("Index already exists in storage/. Use --rebuild to rebuild.")


if __name__ == "__main__":
    main()
Run with --rebuild to force re-indexing.
"""
import argparse
import logging
from pathlib import Path

from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    LLMPredictor,
    load_index_from_storage,
    GPTVectorStoreIndex,
)
from llama_index.embeddings import SentenceTransformerEmbedding
from llama_index.llms import HuggingFaceLLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


STORAGE_DIR = Path("./storage")
RAW_DOCS_DIR = Path("./data/raw_docs")


def build_index(model_embed_name: str = "all-MiniLM-L6-v2", hf_model_name: str = "distilgpt2"):
    logger.info("Building index from docs in %s", RAW_DOCS_DIR)
    documents = SimpleDirectoryReader(str(RAW_DOCS_DIR)).load_data()

    embed_model = SentenceTransformerEmbedding(model_name=model_embed_name)

    # LLM used for concise responses / summaries inside index (local, CPU)
    llm = HuggingFaceLLM(model_name=hf_model_name, task="text-generation", model_kwargs={"temperature": 0.0, "max_length": 256})
    llm_predictor = LLMPredictor(llm=llm)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    logger.info("Index built and persisted to %s", STORAGE_DIR)


def load_index_or_none():
    if not STORAGE_DIR.exists():
        return None
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        index = load_index_from_storage(storage_context)
        return index
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index")
    args = parser.parse_args()

    if args.rebuild:
        build_index()
        return

    existing = load_index_or_none()
    if existing is None:
        logger.info("No existing index found; building now...")
        build_index()
    else:
        logger.info("Index already exists in %s", STORAGE_DIR)


if __name__ == "__main__":
    main()
