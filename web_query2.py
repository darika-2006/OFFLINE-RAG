from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import psycopg2
import os

# ---------------- APP ----------------
app = FastAPI()
# ---------------- DB ----------------
conn = psycopg2.connect(
    dbname="RAGX",
    user="postgres",
    password="Bala@2007",
    host="localhost",
    port="5432"
)
cur = conn.cursor()
# ---------------- MODEL & INDEX ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = 0.45

# if not os.path.exists("halx_index.faiss"):
#     raise RuntimeError("FAISS index not found")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "halx_index.faiss")

def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    return faiss.read_index(FAISS_INDEX_PATH)


index = faiss.read_index("halx_index.faiss")

# ---------------- REQUEST SCHEMA ----------------
class QueryRequest(BaseModel):
    query: str
    role: str


# ---------------- QUERY API ----------------
@app.post("/query")
def query_rag(data: QueryRequest):
    query = data.query.strip()
    user_role = data.role.lower().strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    index = load_faiss_index()
    if index is None:
        return {
            "answer": "No documents have been indexed yet. Please upload documents first.",
            "results": []
        }

    # ---------- FAISS SEARCH ----------
    K_FAISS = 30
    query_embedding = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_embedding, k=K_FAISS)

    # Convert numpy.int64 → int (CRITICAL)
    faiss_chunk_ids = [int(x) for x in I[0]]

    # ---------- DB FILTER (RBAC + ACTIVE) ----------
    sql = """
        SELECT c.chunk_id,c.chunk_text,c.page_number,d.doc_name
        FROM chunks_metadata c
        JOIN documents d ON c.doc_id = d.doc_id
        WHERE c.chunk_id = ANY(%s)
        AND c.is_active = TRUE
        AND c.allowed_roles ILIKE %s
    """
    cur.execute(sql, (faiss_chunk_ids, f"%{user_role}%"))
    rows = cur.fetchall()

    if not rows:
        return {"results": []}

    # Map chunk_id → text
    # chunk_map = {row[0]: row[1] for row in rows}
    chunk_map = {
        row[0]: {
            "text": row[1],
            "page_number": row[2],
            "document": row[3]
        }
        for row in rows
    }
    # Keep FAISS order
    faiss_texts = [chunk_map[cid] for cid in faiss_chunk_ids if cid in chunk_map]

    # Match FAISS distances to filtered chunks
    valid_indices = [
        i for i, cid in enumerate(faiss_chunk_ids) if cid in chunk_map
    ]
    filtered_distances = D[0][valid_indices]

    # ---------- BM25 ----------
    # tokenized_docs = [text.lower().split() for text in faiss_texts]
    tokenized_docs = [chunk["text"].lower().split() for chunk in faiss_texts]
    if not faiss_texts:
        return {"query": query, "role": user_role, "results": []}

    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # ---------- HYBRID SCORING ----------
    # faiss_similarities = 1 / (1 + filtered_distances)
    # combined_scores = (0.7 * faiss_similarities +0.3 * bm25_scores)
    
    faiss_similarities = 1 / (1 + filtered_distances)
    faiss_similarities = np.array(faiss_similarities).flatten()

    if faiss_similarities.size == 0:
        return {
            "query": query,
            "role": user_role,
            "answer": "No relevant information found in the uploaded documents.",
            "results": []
        }

    if float(faiss_similarities.max()) < SIMILARITY_THRESHOLD:
        return {
            "query": query,
            "role": user_role,
            "answer": "The uploaded documents do not contain information relevant to this question.",
            "results": []
        }

    combined_scores = (0.7 * faiss_similarities + 0.3 * bm25_scores)    
    # Rank results
    sorted_indices = np.argsort(combined_scores)[::-1]
    FINAL_K = 3
    final_results = []
    for i in sorted_indices[:FINAL_K]:
        chunk = faiss_texts[i]
        final_results.append({
            "text": chunk["text"],
            "document": chunk["document"],
            "page_number": chunk["page_number"]
    })

    return {
        "query": query,
        "role": user_role,
        "results": final_results
    }