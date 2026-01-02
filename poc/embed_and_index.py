import os
import json
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# -------- Paths --------

BASE_DIR = os.path.dirname(__file__)
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks.json")

EMBED_SAVE = os.path.join(BASE_DIR, "index", "embeddings.npy")
BM25_SAVE = os.path.join(BASE_DIR, "index", "bm25.pkl")

# -------- Load chunks --------

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(texts)} chunks.")

# -------- Embeddings --------

model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nEncoding embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

np.save(EMBED_SAVE, embeddings)
print("Saved embeddings.npy")

# -------- BM25 Index --------

tokenized_texts = [text.lower().split() for text in texts]

bm25 = BM25Okapi(tokenized_texts)

with open(BM25_SAVE, "wb") as f:
    pickle.dump(bm25, f)

print("Saved bm25.pkl")

print("\nIndexing complete! 🚀")
