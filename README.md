# RAG Study Buddy (local)

This project implements a small local Retrieval-Augmented Generation (RAG) "Study Buddy" using:

- LlamaIndex for document indexing & retrieval
- sentence-transformers for embeddings (local)
- a small Hugging Face model (CPU only) for text generation via Transformers
- Gradio for a local chat UI

Everything runs locally. No external web APIs or cloud services are required.

Folders:

- `data/raw_docs/` - put PDFs and .txt files you want to ingest here
- `storage/` - index files persisted here after first build

Quick setup (macOS, zsh):

1. Create and activate a Python virtualenv (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place your PDFs / text files in `data/raw_docs`.

3. Run the app (this will build the index automatically if not present):

```bash
python app.py
```

4. Open the local Gradio URL printed by the script and chat. Answers are generated only from your documents.

Notes:

- The default embedding model is `all-MiniLM-L6-v2` (from sentence-transformers).
- The default generator is `distilgpt2` (small, CPU-friendly). You can change models in `app.py`.
- If you change docs and want to rebuild the index, run:

```bash
python ingest.py --rebuild
```

Troubleshooting:

- If package installation for `faiss-cpu` fails on macOS, you can remove it from `requirements.txt` and use the default LlamaIndex vector store (which may be slower). Alternatively, install faiss via conda: `conda install -c pytorch faiss-cpu`.
