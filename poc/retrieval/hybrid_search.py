import os
import json
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Paths
BASE_DIR = os.path.dirname(__file__)

CHUNKS_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "chunks.json")
EMBED_PATH = os.path.join(BASE_DIR, "..", "index", "embeddings.npy")
BM25_PATH = os.path.join(BASE_DIR, "..", "index", "bm25.pkl")


# Load chunks
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load embeddings
embeddings = np.load(EMBED_PATH)

# Load BM25
with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

# Load encoding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def hybrid_search(query, top_k=5):

    # ----- semantic similarity -----
    query_embedding = model.encode([query])
    semantic_scores = cosine_similarity(query_embedding, embeddings)[0]

    semantic_scores = semantic_scores / semantic_scores.max()


    # ----- keyword score -----
    tokenized_query = query.lower().split()
    keyword_scores = bm25.get_scores(tokenized_query)

    max_val = max(keyword_scores)
    if max_val > 0:
        keyword_scores = keyword_scores / max_val
    else:
        keyword_scores = np.zeros_like(keyword_scores)


    # ----- hybrid total score -----
    combined_scores = semantic_scores + keyword_scores

    top_idxs = combined_scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_idxs:

        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"],
            "score": round(float(combined_scores[idx]),3),
            "access": chunks[idx]["access"]
        })

    return results



# ---------- manual test ----------
if __name__ == "__main__":

    query = input("Enter query: ")
    results = hybrid_search(query)

    for r in results:
        print("\n---")
        print("Chunk:", r["chunk_id"])
        print("Source:", r["source"])
        print("Score:", r["score"])
        print("Text:", r["text"][:120])
