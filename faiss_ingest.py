"""FAISS ingestion script (clean copy).

Use this to build the index: python faiss_ingest.py --rebuild
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
CHUNK_SIZE = 500
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
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def build_index(embed_model_name: str = EMBED_MODEL_NAME):
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No documents found in {RAW_DIR}; add .txt or .pdf files.")

    # force CPU to avoid MPS/multiprocessing issues on macOS
    model = SentenceTransformer(embed_model_name, device="cpu")

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
    (STORAGE_DIR / "metadatas.json").write_text(json.dumps(metadatas, ensure_ascii=False))
    (STORAGE_DIR / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))
    (STORAGE_DIR / "config.json").write_text(json.dumps({"embed_model": embed_model_name}))
    print(f"Built index with {len(texts)} chunks and saved to {STORAGE_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.rebuild or not (STORAGE_DIR.exists() and (STORAGE_DIR / "index.faiss").exists()):
        build_index()
    else:
        print("Index already exists. Use --rebuild to rebuild.")


if __name__ == "__main__":
    main()
