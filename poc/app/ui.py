import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
from retrieval.hybrid_search import hybrid_search
from retrieval.access_filter import filter_by_role
from llm.answer_generator import generate_answer
from utils.pdf_generator import generate_pdf

def pipeline(query, role, progress=gr.Progress()):
    progress(0, desc="🔍 Searching documents...")
    retrieved = hybrid_search(query)

    progress(0.3, desc="🔒 Filtering by access level...")
    filtered = filter_by_role(retrieved, role)

    progress(0.5, desc="🤖 Generating answer...")
    answer = generate_answer(query, filtered)

    progress(0.8, desc="📄 Generating PDF...")
    pdf_path = generate_pdf(answer)

    progress(1.0, desc="✅ Complete!")
    return answer, pdf_path 

ui = gr.Interface(
    fn=pipeline,
    inputs=[
        gr.Textbox(lines=3, label="Question"),
        gr.Radio(["user", "admin"], label="Role", value="user")
    ],
    outputs=[
        gr.Textbox(label="Answer", lines=15),
        gr.File(label="Download PDF")
    ],
    title="RAG System - Powered by Mistral 7B"
)

if __name__ == "__main__":
    ui.launch()