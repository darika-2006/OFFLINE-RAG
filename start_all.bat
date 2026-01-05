@echo off
echo Starting RAG System...

start "Ollama" cmd /k "ollama serve"
timeout /t 3

start "Upload Backend" cmd /k "cd /d D:\RAG\Bala_RAG\OFFLINE-RAG && uvicorn web_text_embedding:app --reload --port 8001"
timeout /t 3

start "Query Backend" cmd /k "cd /d D:\RAG\Bala_RAG\OFFLINE-RAG && uvicorn web_query:app --reload --port 8000"
timeout /t 3

start "Frontend" cmd /k "cd /d D:\RAG\Bala_RAG\OFFLINE-RAG\app && streamlit run main_app.py"

echo All services started!
pause