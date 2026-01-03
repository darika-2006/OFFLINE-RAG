from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import psycopg2

conn = psycopg2.connect( 
    dbname="RAG",
    user="postgres",
    password="Bala@2007",
    host="localhost",
    port="5432"
)
cur=conn.cursor()

query=input("Enter the query you want to retieve : ")
user_role = input("Enter your role : ").lower()

tokenized_query = query.lower().split()

index = faiss.read_index("hal_index.faiss")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

K_FAISS = 10
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=K_FAISS)

faiss_chunk_ids = [int(x) for x in I[0]]

sql = """
SELECT chunk_id, chunk_text
FROM chunks_metadata
WHERE chunk_id = ANY(%s)
  AND is_active = TRUE
  AND allowed_roles ILIKE %s"""
cur.execute(sql, (faiss_chunk_ids, f"%{user_role}%"))

rows = cur.fetchall()

# Map chunk_id → text
chunk_map = {row[0]: row[1] for row in rows}
faiss_texts = [chunk_map[cid] for cid in faiss_chunk_ids if cid in chunk_map]

# Keep FAISS distances only for chunks that survived DB/RBAC filtering
valid_indices = [i for i, cid in enumerate(faiss_chunk_ids) if cid in chunk_map]
filtered_faiss_distances = D[0][valid_indices]


tokenized_faiss_texts = [text.lower().split() for text in faiss_texts]

bm25_local = BM25Okapi(tokenized_faiss_texts)
tokenized_query = query.lower().split()
bm25_scores = bm25_local.get_scores(tokenized_query)
bm25_scores = np.array(bm25_scores)
if bm25_scores.max() > 0:
    bm25_scores = bm25_scores / bm25_scores.max()

faiss_distances = D[0]
faiss_similarities = 1 / (1 + filtered_faiss_distances)
combined_scores = []

for i in range(len(faiss_similarities)):
    score = 0.7 * faiss_similarities[i] + 0.3 * bm25_scores[i]
    combined_scores.append(score)
sorted_indices = sorted(
    range(len(combined_scores)),
    key=lambda i: combined_scores[i],
    reverse=True
)
FINAL_K = 3
final_results = [faiss_texts[i] for i in sorted_indices[:FINAL_K]]
if not final_results:
    print("No relevant information found.")
else:
    for text in final_results:
        print("\n---")
        print(text)
