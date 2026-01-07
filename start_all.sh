#!/bin/bash

echo "Starting RAG System..."

# Start Ollama
gnome-terminal -- bash -c "ollama serve; exec bash" &
sleep 3

# Start Upload Backend
gnome-terminal -- bash -c "cd /home/darika-sivakumar/IIIT_DHARWAD/OFFLINE-RAG && uvicorn web_text_embedding:app --reload --port 8001; exec bash" &
sleep 3

# Start Query Backend
gnome-terminal -- bash -c "cd /home/darika-sivakumar/IIIT_DHARWAD/OFFLINE-RAG && uvicorn web_query:app --reload --port 8000; exec bash" &
sleep 3

# Start Frontend
gnome-terminal -- bash -c "cd /home/darika-sivakumar/IIIT_DHARWAD/OFFLINE-RAG/app && streamlit run main_app.py; exec bash" &

echo "All services started!"