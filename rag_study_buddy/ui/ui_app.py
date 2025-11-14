"""Gradio UI builder for the RAG Study Buddy.

Exposes a `build_ui(pipeline)` function returning started Gradio Blocks app
objects. The UI supports asking questions and triggering a background
rebuild which will automatically reload the in-memory index when finished.
"""
from __future__ import annotations

import os
import subprocess
import threading
from typing import Callable

import gradio as gr


def _run_rebuild_in_thread(on_finished: Callable[[], None], log_path: str = "./storage/rebuild.log") -> None:
    """Run the ingest script in a subprocess and call on_finished() when done.

    Output (stdout/stderr) is written to log_path for debugging.
    """
    def _worker():
        with open(log_path, "w", encoding="utf-8") as fh:
            # Use the same python executable to run the package module
            cmd = [os.sys.executable, "-m", "rag_study_buddy.ingest", "--input_dir", "./data/raw_docs", "--persist_dir", "./data/index"]
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            proc.wait()
        # rebuild finished
        try:
            on_finished()
        except Exception:
            # swallow errors in callback
            pass

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def build_ui(pipeline, title: str = "RAG Study Buddy") -> gr.Blocks:
    """Create a Gradio Blocks UI.

    `pipeline` must implement .answer(question) -> (answer, sources) and
    .reload_index() to reload after rebuilds.
    """
    with gr.Blocks(title=title, css=".chatbox {min-height: 300px}") as demo:
        gr.Markdown(f"## {title}")

        with gr.Row():
            chat = gr.Chatbot(elem_id="chatbot")
            with gr.Column(scale=0.6):
                question = gr.Textbox(lines=2, placeholder="Ask a question about your docs...")
                btn = gr.Button("Send")
                rebuild_btn = gr.Button("Rebuild Index")
                status = gr.Textbox(value="Index loaded", label="Status", interactive=False)

        def _submit(q, history):
            if not q or q.strip() == "":
                return history, ""
            answer, sources = pipeline.answer(q)
            history = history + [(q, answer)]
            status.value = f"Last query used {len(sources)} sources"
            return history, ""

        btn.click(fn=_submit, inputs=[question, chat], outputs=[chat, question])

        def _start_rebuild():
            status.value = "Rebuilding index... (see ./storage/rebuild.log)"

            def _on_done():
                # reload the pipeline index in memory
                try:
                    pipeline.reload_index()
                    status.value = "Rebuild finished — index reloaded"
                except Exception as e:
                    status.value = f"Rebuild finished but reload failed: {e}"

            _run_rebuild_in_thread(on_finished=_on_done, log_path="./storage/rebuild.log")
            return "Rebuild started"

        rebuild_btn.click(fn=_start_rebuild, inputs=[], outputs=[status])

    return demo
