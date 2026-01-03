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
    dbname="RAG",
    user="postgres",
    password="Bala@2007",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# ---------------- MODEL & INDEX ----------------
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

if not os.path.exists("hal_index.faiss"):
    raise RuntimeError("FAISS index not found")

index = faiss.read_index("hal_index.faiss")

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

    # ---------- FAISS SEARCH ----------
    K_FAISS = 10
    query_embedding = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_embedding, k=K_FAISS)

    # Convert numpy.int64 → int (CRITICAL)
    faiss_chunk_ids = [int(x) for x in I[0]]

    # ---------- DB FILTER (RBAC + ACTIVE) ----------
    sql = """
    SELECT chunk_id, chunk_text
    FROM chunks_metadata
    WHERE chunk_id = ANY(%s)
      AND is_active = TRUE
      AND allowed_roles ILIKE %s
    """
    cur.execute(sql, (faiss_chunk_ids, f"%{user_role}%"))
    rows = cur.fetchall()

    if not rows:
        return {"results": []}

    # Map chunk_id → text
    chunk_map = {row[0]: row[1] for row in rows}

    # Keep FAISS order
    faiss_texts = [chunk_map[cid] for cid in faiss_chunk_ids if cid in chunk_map]

    # Match FAISS distances to filtered chunks
    valid_indices = [
        i for i, cid in enumerate(faiss_chunk_ids) if cid in chunk_map
    ]
    filtered_distances = D[0][valid_indices]

    # ---------- BM25 ----------
    tokenized_docs = [text.lower().split() for text in faiss_texts]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # ---------- HYBRID SCORING ----------
    faiss_similarities = 1 / (1 + filtered_distances)

    combined_scores = (
        0.7 * faiss_similarities +
        0.3 * bm25_scores
    )

    # Rank results
    sorted_indices = np.argsort(combined_scores)[::-1]

    FINAL_K = 3
    final_results = [
        faiss_texts[i] for i in sorted_indices[:FINAL_K]
    ]

    return {
        "query": query,
        "role": user_role,
        "results": final_results
    }
