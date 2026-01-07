from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import psycopg2
import os
import ollama
import pickle
import torch
import json

# ---------------- APP ----------------
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DB ----------------
conn = psycopg2.connect(
    dbname="RAG",
    user="postgres",
    password="darika_2006",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# ---------------- MODEL & INDEX ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Embedding Device: {device}")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                            device = device)

# Load FAISS index
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "hal_index.faiss")
MAPPING_PATH = os.path.join(INDEX_DIR, "chunk_id_mapping.pkl")

if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
else:
    print(f"⚠️ FAISS index not found")
    index = faiss.IndexFlatL2(768)
    faiss.write_index(index, FAISS_INDEX_PATH)

# Load mapping
if os.path.exists(MAPPING_PATH):
    with open(MAPPING_PATH, 'rb') as f:
        chunk_id_mapping = pickle.load(f)
    print(f"✅ Loaded chunk_id mapping with {len(chunk_id_mapping)} entries")
else:
    print(f"⚠️ Mapping not found. Creating empty mapping.")
    chunk_id_mapping = []

# Verify alignment
if len(chunk_id_mapping) != index.ntotal:
    print(f"⚠️ WARNING: Mapping size ({len(chunk_id_mapping)}) != FAISS size ({index.ntotal})")

# ---------------- ROOT ENDPOINT ----------------
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Query Service",
        "faiss_index_size": index.ntotal,
        "mapping_size": len(chunk_id_mapping),
        "alignment_ok": len(chunk_id_mapping) == index.ntotal
    }

# ---------------- REQUEST SCHEMA ----------------
class QueryRequest(BaseModel):
    query: str
    role: str

# ---------------- LLM FUNCTION WITH AUTO GPU/CPU DETECTION ----------------
def generate_answer(query: str, context_chunks: list) -> str:
    """Generate answer using Ollama - Auto GPU/CPU fallback"""
    if not context_chunks:
        return "No relevant information found in the knowledge base."
    
    # Limit context
    context = "\n\n---\n\n".join(context_chunks[:4])[:4500]
    
    prompt = f"""Using the context below, explain the concept clearly in your own words.
If the context lists topics or references, infer the general meaning:

Context:
{context}

Question: {query}

Answer (5-6 sentences):"""
    
    try:
        import time
        start = time.time()
        
        # Detect GPU availability
        has_gpu = False
        
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"   🎮 GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Try GPU first with memory limit
            try:
                response = ollama.chat(
                    model="phi3:mini",
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "num_predict": 150,
                        "temperature": 0.2,
                        "num_gpu": 0,
                        "num_thread": 6,           # Use GPU
                        "num_ctx": 1024,  
                    }
                )
                elapsed = time.time() - start
                print(f"   ⏱️ LLM: {elapsed:.2f}s (GPU mode)")
                return response['message']['content']
                
            except Exception as gpu_error:
                print(f"   ⚠️ GPU failed: {str(gpu_error)}")
                print(f"   🔄 Falling back to CPU...")
                # Fall through to CPU mode below
        
        # CPU Mode (either no GPU or GPU failed)
        print(f"   💻 Using CPU mode")
        response = ollama.chat(
            model="phi3:mini",
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": 256,          # Shorter for CPU
                "temperature": 0.2,
                "num_gpu": 0,               # Force CPU
                "num_thread": 8,            # CPU threads
                "num_ctx": 2048             # Smaller context for CPU
            }
        )
        
        elapsed = time.time() - start
        print(f"   ⏱️ LLM: {elapsed:.2f}s (CPU mode)")
        return response['message']['content']
        
    except Exception as e:
        print(f"❌ LLM Error: {str(e)}")
        
        # Check if Ollama is running
        try:
            import requests
            requests.get("http://localhost:11434", timeout=1)
            ollama_status = "running but model failed"
        except:
            ollama_status = "NOT RUNNING - Start with: ollama serve"
        
        return f"""**LLM Error:** {ollama_status}

**Relevant excerpt from documents:**

{context_chunks[0][:600]}...

---
*To enable AI answers, ensure Ollama is running: `ollama serve`*"""

# ---------------- QUERY API ----------------
@app.post("/query")
def query_rag(data: QueryRequest):
    import time
    start_time = time.time()
    
    query = data.query.strip()
    user_role = data.role.lower().strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    print(f"\n🔍 Query: '{query}' (role: {user_role})")

    # Check if index has vectors
    if index.ntotal == 0:
        return {
            "query": query,
            "role": user_role,
            "answer": "No documents uploaded yet.",
            "sources": []
        }

    # ---------- FAISS SEARCH ----------
    t1 = time.time()
    K_FAISS = 15
    query_embedding = model.encode([query], normalize_embeddings=True)
    encode_time = time.time() - t1

    t2 = time.time()
    D, I = index.search(query_embedding, k=min(K_FAISS, index.ntotal))
    faiss_time = time.time() - t2

    print(f"   ⏱️ Encode: {encode_time:.2f}s | FAISS: {faiss_time:.2f}s")

    # Convert FAISS positions to chunk_ids using mapping
    faiss_positions = [int(x) for x in I[0] if x != -1]
    faiss_chunk_ids = [chunk_id_mapping[pos] for pos in faiss_positions if pos < len(chunk_id_mapping)]
    
    print(f"   📍 Positions: {faiss_positions}")
    print(f"   🆔 Chunk IDs: {faiss_chunk_ids}")

    if not faiss_chunk_ids:
        return {
            "query": query,
            "role": user_role,
            "answer": "No relevant documents found.",
            "sources": []
        }

    # ---------- DB FILTER ----------
    t2 = time.time()
    placeholders = ",".join(["%s"] * len(faiss_chunk_ids))
    sql = f"""
    SELECT chunk_id, chunk_text
    FROM chunks_metadata
    WHERE chunk_id IN ({placeholders})
      AND is_active = TRUE
      AND (allowed_roles LIKE %s OR allowed_roles = 'public')
    """
    
    role_pattern = f"%{user_role}%"
    cur.execute(sql, (*faiss_chunk_ids, role_pattern))
    rows = cur.fetchall()
    print(f"   ⏱️ DB Filter: {time.time()-t2:.2f}s ({len(rows)} accessible)")

    if not rows:
        return {
            "query": query,
            "role": user_role,
            "answer": f"No documents for role '{user_role}'.",
            "sources": []
        }

    filtered_chunks = {row[0]: row[1] for row in rows}

    # ---------- BM25 RE-RANKING ----------
    t3 = time.time()
    chunk_texts = list(filtered_chunks.values())
    tokenized_chunks = [text.lower().split() for text in chunk_texts]
    
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.lower().split() 
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top 5
    top_indices = np.argsort(bm25_scores)[-5:][::-1]
    top_chunks = [chunk_texts[i] for i in top_indices]
    print(f"   ⏱️ BM25: {time.time()-t3:.2f}s")

    # ---------- GENERATE ANSWER ----------
    answer = generate_answer(query, top_chunks)

    total_time = time.time() - start_time
    print(f"   ✅ Total: {total_time:.2f}s\n")

    return {
        "query": query,
        "role": user_role,
        "answer": answer,
        "sources": top_chunks
    }

# ---------------- STREAMING QUERY API ----------------
@app.post("/query/stream")
async def query_rag_stream(data: QueryRequest):
    """Streaming version of query endpoint"""
    import time
    
    async def generate():
        query = data.query.strip()
        user_role = data.role.lower().strip()
        
        # Send initial status
        yield f"data: {json.dumps({'status': 'searching', 'message': 'Searching documents...'})}\n\n"
        
        # Check if index has vectors
        if index.ntotal == 0:
            yield f"data: {json.dumps({'status': 'complete', 'message': 'No documents uploaded yet.', 'answer': '', 'sources': []})}\n\n"
            return
        
        # ---------- FAISS SEARCH ----------
        t1 = time.time()
        K_FAISS = 15
        query_embedding = model.encode([query], normalize_embeddings=True)
        encode_time = time.time() - t1

        t2 = time.time()
        D, I = index.search(query_embedding, k=min(K_FAISS, index.ntotal))
        faiss_time = time.time() - t2
        print(f"   ⏱️ Encode: {encode_time:.2f}s | FAISS: {faiss_time:.2f}s")
        
        # Convert FAISS positions to chunk_ids using mapping
        faiss_positions = [int(x) for x in I[0] if x != -1]
        faiss_chunk_ids = [chunk_id_mapping[pos] for pos in faiss_positions if pos < len(chunk_id_mapping)]
        
        print(f"   ⏱️ FAISS: {time.time()-t1:.2f}s")
        print(f"   📍 Positions: {faiss_positions}")
        print(f"   🆔 Chunk IDs: {faiss_chunk_ids}")

        if not faiss_chunk_ids:
            yield f"data: {json.dumps({'status': 'complete', 'message': 'No relevant documents found.', 'answer': '', 'sources': []})}\n\n"
            return

        # ---------- DB FILTER ----------
        t2 = time.time()
        placeholders = ",".join(["%s"] * len(faiss_chunk_ids))
        sql = f"""
        SELECT chunk_id, chunk_text
        FROM chunks_metadata
        WHERE chunk_id IN ({placeholders})
          AND is_active = TRUE
          AND (allowed_roles LIKE %s OR allowed_roles = 'public')
        """
        
        role_pattern = f"%{user_role}%"
        cur.execute(sql, (*faiss_chunk_ids, role_pattern))
        rows = cur.fetchall()
        print(f"   ⏱️ DB Filter: {time.time()-t2:.2f}s ({len(rows)} accessible)")

        if not rows:
            yield f"data: {json.dumps({'status': 'complete', 'message': f'No documents for role \'{user_role}\'.', 'answer': '', 'sources': []})}\n\n"
            return

        filtered_chunks = {row[0]: row[1] for row in rows}

        # ---------- BM25 RE-RANKING ----------
        t3 = time.time()
        chunk_texts = list(filtered_chunks.values())
        tokenized_chunks = [text.split() for text in chunk_texts]
        
        bm25 = BM25Okapi(tokenized_chunks)
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Get top 3
        top_indices = np.argsort(bm25_scores)[-3:][::-1]
        top_chunks = [chunk_texts[i] for i in top_indices]
        print(f"   ⏱️ BM25: {time.time()-t3:.2f}s")

        # Send found status
        yield f"data: {json.dumps({'status': 'found', 'message': f'Found {len(top_chunks)} relevant chunks'})}\n\n"
        
        # Generate answer
        yield f"data: {json.dumps({'status': 'generating', 'message': 'Generating answer (may take 1-2 minutes)...'})}\n\n"
        
        answer = generate_answer(query, top_chunks)
        
        # Send final result
        result = {
            "status": "complete",
            "query": query,
            "role": user_role,
            "answer": answer,
            "sources": top_chunks
        }
        yield f"data: {json.dumps(result)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ---------------- HEALTH CHECK ENDPOINT FOR LLM ----------------
@app.get("/health/llm")
def check_llm_health():
    """Check if Ollama LLM is available"""
    try:
        import requests
        
        # Check Ollama service
        r = requests.get("http://localhost:11434", timeout=2)
        ollama_running = r.status_code == 200
        
        # Check GPU
        has_gpu = torch.cuda.is_available()
        gpu_info = None
        if has_gpu:
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                "memory_free_gb": round(torch.cuda.mem_get_info()[0] / 1024**3, 2)
            }
        
        # Test LLM
        llm_working = False
        llm_mode = "unknown"
        
        if ollama_running:
            try:
                response = ollama.chat(
                    model="phi3:mini",
                    messages=[{"role": "user", "content": "test"}],
                    options={"num_predict": 5}
                )
                llm_working = True
                llm_mode = "GPU" if has_gpu else "CPU"
            except Exception as e:
                llm_working = False
                llm_mode = f"error: {str(e)[:100]}"
        
        return {
            "ollama_service": "running" if ollama_running else "offline",
            "llm_working": llm_working,
            "llm_mode": llm_mode,
            "gpu_available": has_gpu,
            "gpu_info": gpu_info,
            "recommendation": "Start Ollama: ollama serve" if not ollama_running else "OK"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "ollama_service": "offline",
            "recommendation": "Start Ollama with: ollama serve"
        }
