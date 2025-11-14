"""Top-level runner for the RAG Study Buddy app.

Runs the Gradio app and wires the pipeline to the UI.
"""
from rag_study_buddy.rag_pipeline import RAGPipeline
from rag_study_buddy.ui.ui_app import build_ui


def main():
    pipeline = RAGPipeline(persist_dir="./data/index", gen_model="distilgpt2")
    app = build_ui(pipeline, title="RAG Study Buddy")
    app.launch()


if __name__ == "__main__":
    main()
"""Top-level runner for the RAG Study Buddy app.

Runs the Gradio app and wires the pipeline to the UI.
"""
from rag_study_buddy.rag_pipeline import RAGPipeline
from rag_study_buddy.ui.ui_app import build_ui


def main():
    pipeline = RAGPipeline(persist_dir="./data/index", gen_model="distilgpt2")
    app = build_ui(pipeline, title="RAG Study Buddy")
    app.launch()


if __name__ == "__main__":
    main()
"""Run the local RAG Study Buddy with a Gradio UI using sentence-transformers + FAISS + transformers.

Features:
- aesthetic Gradio chat UI
- background rebuild of FAISS index (runs `faiss_ingest.py --rebuild`)
- automatic reload of index in memory after rebuild completes

Everything runs locally (CPU).
"""
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROOT = Path(__file__).parent
RAW_DOCS = ROOT / "data" / "raw_docs"
STORAGE_DIR = ROOT / "storage"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GEN_MODEL = "distilgpt2"

# caches
_embed_models: Dict[str, SentenceTransformer] = {}
_gen_models: Dict[str, Dict[str, Any]] = {}
_index_cache: Optional[faiss.Index] = None
_texts_cache: Optional[List[str]] = None
_metas_cache: Optional[List[Dict[str, Any]]] = None
_cfg_cache: Dict[str, Any] = {}

# CPU-only safety for macOS (prevent MPS multiprocessing segfaults)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)
DEVICE = torch.device("cpu")


def reload_storage() -> str:
    """Load index, texts, metas, and config from disk into memory caches.

    Returns a status string for logging.
    """
    global _index_cache, _texts_cache, _metas_cache, _cfg_cache
    idx_path = STORAGE_DIR / "index.faiss"
    texts_path = STORAGE_DIR / "texts.json"
    metas_path = STORAGE_DIR / "metadatas.json"
    cfg_path = STORAGE_DIR / "config.json"

    if not idx_path.exists():
        return f"Index file not found at {idx_path}"

    try:
        index = faiss.read_index(str(idx_path))
        texts = json.loads(texts_path.read_text(encoding="utf-8")) if texts_path.exists() else []
        metas = json.loads(metas_path.read_text(encoding="utf-8")) if metas_path.exists() else []
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

        _index_cache = index
        _texts_cache = texts
        _metas_cache = metas
        _cfg_cache = cfg
        return f"Reloaded index ({len(texts)} chunks)"
    except Exception as e:
        logger.exception("Failed to reload storage: %s", e)
        return f"Failed to reload storage: {e}"


def get_cached_storage():
    """Return cached index/texts/metas or raise if not loaded."""
    if _index_cache is None or _texts_cache is None or _metas_cache is None:
        # attempt a lazy reload
        msg = reload_storage()
        logger.info(msg)
        if _index_cache is None:
            raise RuntimeError("Index not loaded. Run rebuild or check storage directory.")
    return _index_cache, _texts_cache, _metas_cache, _cfg_cache


def get_answer(query: str, top_k: int = 4, embed_model_name: str = DEFAULT_EMBED_MODEL, gen_model_name: str = DEFAULT_GEN_MODEL) -> Dict[str, Any]:
    index, texts, metas, cfg = get_cached_storage()

    # embed model (cached)
    embed_model = _embed_models.get(embed_model_name)
    if embed_model is None:
        embed_model = SentenceTransformer(embed_model_name, device="cpu")
        _embed_models[embed_model_name] = embed_model

    q_emb = embed_model.encode([query], convert_to_numpy=True)

    D, I = index.search(q_emb, top_k)
    indices = I[0].tolist()

    retrieved_texts = [texts[i] for i in indices if i < len(texts)]
    retrieved_metas = [metas[i] for i in indices if i < len(metas)]

    context = "\n---\n".join(retrieved_texts)

    prompt = f"Use the following context to answer the question. Be concise and answer only from the context.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"

    # generator (tokenizer+model) cached
    gen_entry = _gen_models.get(gen_model_name)
    if gen_entry is None:
        tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        model = AutoModelForCausalLM.from_pretrained(gen_model_name)
        model.to(DEVICE)
        model.eval()
        gen_entry = {"tokenizer": tokenizer, "model": model}
        _gen_models[gen_model_name] = gen_entry

    tokenizer = gen_entry["tokenizer"]
    model = gen_entry["model"]

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE) if hasattr(inputs, "attention_mask") else None

    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=150, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # trim to the answer part after the prompt
    if prompt in answer:
        answer = answer.split(prompt, 1)[-1].strip()

    sources = []
    for m, t in zip(retrieved_metas, retrieved_texts):
        sources.append({"source": m.get("source"), "snippet": (t[:300] + "...") if len(t) > 300 else t})

    return {"answer": answer, "sources": sources}


def start_rebuild_in_background(python_exe: str) -> None:
    """Start faiss_ingest.py --rebuild in a background thread and reload when done."""

    def _run():
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        log_path = STORAGE_DIR / "rebuild.log"
        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.Popen([python_exe, str(ROOT / "faiss_ingest.py"), "--rebuild"], cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # stream stdout
            if proc.stdout is not None:
                for line in proc.stdout:
                    f.write(line)
                    f.flush()
            proc.wait()
            f.write(f"\nPROCESS EXITED: {proc.returncode}\n")
            f.flush()

        # after process completes, attempt to reload index into memory
        reload_msg = reload_storage()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\nRELOAD: {reload_msg}\n")

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def build_ui():
    css = """
    body { background: #0f1724 }
    .chatbox { background: linear-gradient(180deg,#0b1220 0%,#071022 100%); border-radius:12px; padding:12px }
    .left-panel { background: transparent }
    .gradio-container { color: #dbeafe }
    .title { font-family: 'Inter', sans-serif; color: #ecfeff }
    .subtitle { color: #9ca3af }
    .controls { background: #071029; border-radius:10px; padding:12px }
    """

    with gr.Blocks(css=css, theme=None) as demo:
        with gr.Row():
            with gr.Column(scale=3, elem_classes="left-panel"):
                gr.HTML("<h1 class='title'>RAG Study Buddy</h1><p class='subtitle'>Local retrieval-powered assistant — answers only from your docs.</p>")
                chatbot = gr.Chatbot(elem_id="chatbot", label="Conversation")
                with gr.Row():
                    txt = gr.Textbox(placeholder="Ask a question about your documents...", show_label=False, lines=2)
                    send = gr.Button("Send")
                with gr.Row():
                    clear = gr.Button("Clear Chat")
            with gr.Column(scale=1):
                gr.HTML("<div class='controls'><h3 style='margin-top:0'>Controls</h3></div>")
                model_gen = gr.Dropdown(choices=[DEFAULT_GEN_MODEL], value=DEFAULT_GEN_MODEL, label="Generator model")
                model_embed = gr.Dropdown(choices=[DEFAULT_EMBED_MODEL], value=DEFAULT_EMBED_MODEL, label="Embedder model")
                top_k = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Top-k passages")
                rebuild_btn = gr.Button("Rebuild index")
                refresh_btn = gr.Button("Refresh status")
                rebuild_status = gr.Textbox(label="Rebuild log", lines=8)
                src = gr.Dataframe(headers=["source", "snippet"], label="Top source snippets", interactive=False)

        # state: history list and last sources
        state_chat = gr.State([])

        def send_message(message, chat_history, embed_m, gen_m, k):
            if not message or not message.strip():
                return chat_history, []
            # append user message
            chat_history = chat_history + [("You", message)]
            res = get_answer(message, top_k=int(k), embed_model_name=embed_m, gen_model_name=gen_m)
            answer = res["answer"]
            sources = res["sources"]
            chat_history = chat_history + [("StudyBuddy", answer)]
            # format sources for dataframe
            tab = []
            for s in sources:
                tab.append([s.get("source"), s.get("snippet")])
            return chat_history, tab

        send = gr.Button.update if False else None
        send = gr.Button  # satisfy some linters; actual widget assigned above

        def clear_chat():
            return [], []

        send_button = demo.get_component_by_id("Send") if False else None

        send.click(send_message, inputs=[txt, state_chat, model_embed, model_gen, top_k], outputs=[chatbot, src])
        txt.submit(send_message, inputs=[txt, state_chat, model_embed, model_gen, top_k], outputs=[chatbot, src])
        clear.click(lambda: ([], []), outputs=[chatbot, src])

        def on_rebuild_click():
            start_rebuild_in_background(sys.executable)
            return "Rebuild started. Click Refresh to load progress."

        def on_refresh_click():
            log_path = STORAGE_DIR / "rebuild.log"
            if not log_path.exists():
                return "No rebuild log found. Start a rebuild to generate a log."
            return log_path.read_text(encoding="utf-8")

        rebuild_btn.click(lambda: on_rebuild_click(), outputs=[rebuild_status])
        refresh_btn.click(lambda: on_refresh_click(), outputs=[rebuild_status])

    return demo


def main():
    # try to load storage on startup (best-effort)
    try:
        msg = reload_storage()
        logger.info(msg)
    except Exception as e:
        logger.info("Startup reload failed: %s", e)

    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
"""Run the local RAG Study Buddy with a Gradio UI using sentence-transformers + FAISS + transformers.

This implementation avoids LlamaIndex and runs fully locally (CPU).
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import os
import faiss
import gradio as gr
import threading
import subprocess
import sys
from pathlib import Path as P
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


"""Run the local RAG Study Buddy with a Gradio UI using sentence-transformers + FAISS + transformers.

This implementation avoids LlamaIndex and runs fully locally (CPU).
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RAW_DOCS = Path("./data/raw_docs")
STORAGE_DIR = Path("./storage")
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GEN_MODEL = "distilgpt2"

# simple in-memory caches so we don't reload models on every query
_embed_models: Dict[str, SentenceTransformer] = {}
_gen_models: Dict[str, Dict[str, Any]] = {}

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)
DEVICE = torch.device("cpu")


def load_storage():
    idx_path = STORAGE_DIR / "index.faiss"
    texts_path = STORAGE_DIR / "texts.json"
    metas_path = STORAGE_DIR / "metadatas.json"
    cfg_path = STORAGE_DIR / "config.json"
    if not idx_path.exists():
        raise RuntimeError(f"Index not found at {idx_path}. Run `python faiss_ingest.py --rebuild` first.")
    index = faiss.read_index(str(idx_path))
    texts = json.loads(texts_path.read_text(encoding="utf-8"))
    metas = json.loads(metas_path.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    return index, texts, metas, cfg


def get_answer(query: str, top_k: int = 4, embed_model_name: str = DEFAULT_EMBED_MODEL, gen_model_name: str = DEFAULT_GEN_MODEL) -> Dict[str, Any]:
    index, texts, metas, cfg = load_storage()

    # embed model (cached)
    embed_model = _embed_models.get(embed_model_name)
    if embed_model is None:
        # force CPU to avoid MPS / multiproc issues on macOS
        embed_model = SentenceTransformer(embed_model_name, device="cpu")
        _embed_models[embed_model_name] = embed_model

    q_emb = embed_model.encode([query], convert_to_numpy=True)

    D, I = index.search(q_emb, top_k)
    indices = I[0].tolist()

    retrieved_texts = [texts[i] for i in indices if i < len(texts)]
    retrieved_metas = [metas[i] for i in indices if i < len(metas)]

    context = "\n---\n".join(retrieved_texts)

    prompt = f"Use the following context to answer the question. Be concise and answer only from the context.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"

    # generator (tokenizer+model) cached
    gen_entry = _gen_models.get(gen_model_name)
    if gen_entry is None:
        tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        model = AutoModelForCausalLM.from_pretrained(gen_model_name)
        model.to(DEVICE)
        model.eval()
        gen_entry = {"tokenizer": tokenizer, "model": model}
        _gen_models[gen_model_name] = gen_entry

    tokenizer = gen_entry["tokenizer"]
    model = gen_entry["model"]

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE) if hasattr(inputs, "attention_mask") else None

    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=150, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # trim to the answer part after the prompt
    if prompt in answer:
        answer = answer.split(prompt, 1)[-1].strip()

    sources = []
    for m, t in zip(retrieved_metas, retrieved_texts):
        sources.append({"source": m.get("source"), "snippet": (t[:300] + "...") if len(t) > 300 else t})

    return {"answer": answer, "sources": sources}


def main():
    def _ask(q: str):
        if not q or not q.strip():
            return "", []
        res = get_answer(q)
        return res["answer"], res["sources"]

    # --- UI: aesthetic chat layout ---
    css = """
    body { background: #0f1724 }
    .chatbox { background: linear-gradient(180deg,#0b1220 0%,#071022 100%); border-radius:12px; padding:12px }
    .left-panel { background: transparent }
    .gradio-container { color: #dbeafe }
    .title { font-family: 'Inter', sans-serif; color: #ecfeff }
    .subtitle { color: #9ca3af }
    .controls { background: #071029; border-radius:10px; padding:12px }
    """

    with gr.Blocks(css=css, theme=None) as demo:
        with gr.Row():
            with gr.Column(scale=3, elem_classes="left-panel"):
                gr.HTML("<h1 class='title'>RAG Study Buddy</h1><p class='subtitle'>Local retrieval-powered assistant — answers only from your docs.</p>")
                chatbot = gr.Chatbot(elem_id="chatbot", label="Conversation")
                with gr.Row():
                    txt = gr.Textbox(placeholder="Ask a question about your documents...", show_label=False, lines=2)
                    send = gr.Button("Send")
                with gr.Row():
                    clear = gr.Button("Clear Chat")
            with gr.Column(scale=1):
                gr.HTML("<div class='controls'><h3 style='margin-top:0'>Controls</h3></div>")
                model_gen = gr.Dropdown(choices=[DEFAULT_GEN_MODEL], value=DEFAULT_GEN_MODEL, label="Generator model")
                model_embed = gr.Dropdown(choices=[DEFAULT_EMBED_MODEL], value=DEFAULT_EMBED_MODEL, label="Embedder model")
                top_k = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Top-k passages")
                    rebuild_btn = gr.Button("Rebuild index")
                    refresh_btn = gr.Button("Refresh status")
                    rebuild_status = gr.Textbox(label="Rebuild log", lines=8)
                src = gr.Dataframe(headers=["source", "snippet"], label="Top source snippets", interactive=False)
        # rebuild helpers
        def _run_rebuild(python_exe: str):
            log_path = STORAGE_DIR / "rebuild.log"
            # ensure storage dir exists
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                proc = subprocess.Popen([python_exe, "faiss_ingest.py", "--rebuild"], cwd=str(P.cwd()), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    f.write(line)
                    f.flush()

        def rebuild_index_click():
            # start background thread to run rebuild with current python executable
            python_exe = sys.executable
            t = threading.Thread(target=_run_rebuild, args=(python_exe,), daemon=True)
            t.start()
            return "Rebuild started. Click Refresh to load progress."

        def refresh_status_click():
            log_path = STORAGE_DIR / "rebuild.log"
            if not log_path.exists():
                return "No rebuild log found. Start a rebuild to generate a log."
            return log_path.read_text(encoding="utf-8")

        rebuild_btn.click(rebuild_index_click, outputs=[rebuild_status])
        refresh_btn.click(refresh_status_click, outputs=[rebuild_status])

        # state: history list and last sources
        state_chat = gr.State([])
        state_sources = gr.State([])

        def send_message(message, chat_history, embed_m, gen_m, k):
            if not message or not message.strip():
                return chat_history, []
            # append user message
            chat_history = chat_history + [("You", message)]
            res = get_answer(message, top_k=int(k), embed_model_name=embed_m, gen_model_name=gen_m)
            answer = res["answer"]
            sources = res["sources"]
            chat_history = chat_history + [("StudyBuddy", answer)]
            # format sources for dataframe
            tab = []
            for s in sources:
                tab.append([s.get("source"), s.get("snippet")])
            return chat_history, tab

        def clear_chat():
            return [], []

        send.click(send_message, inputs=[txt, state_chat, model_embed, model_gen, top_k], outputs=[chatbot, src])
        txt.submit(send_message, inputs=[txt, state_chat, model_embed, model_gen, top_k], outputs=[chatbot, src])
        clear.click(lambda: ([], []), outputs=[chatbot, src])

    demo.launch()


if __name__ == "__main__":
    main()
