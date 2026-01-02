import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from retrieval.hybrid_search import hybrid_search
from retrieval.access_filter import filter_by_role
from llm.answer_generator import generate_answer

def pipeline(query, role, progress=gr.Progress()):
    
    progress(0, desc="🔍 Searching documents...")
    retrieved = hybrid_search(query)
    
    progress(0.3, desc="🔒 Filtering by access level...")
    filtered = filter_by_role(retrieved, role)
    
    progress(0.5, desc="🤖 Generating answer with Mistral 7B (10-30 seconds)...")
    answer = generate_answer(query, filtered)
    
    progress(1.0, desc="✅ Complete!")
    return answer

ui = gr.Interface(
    fn=pipeline,
    inputs=[
        gr.Textbox(lines=3, label="Question", placeholder="Ask your question here..."),
        gr.Radio(["user", "admin"], label="Role", value="user")
    ],
    outputs=gr.Textbox(label="Answer", lines=15, max_lines=25),
    title="RAG System - Powered by Mistral 7B"
)

ui.launch(theme=gr.themes.Soft())