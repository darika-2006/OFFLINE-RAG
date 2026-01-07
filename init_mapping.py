import psycopg2
import pickle
import os

conn = psycopg2.connect(
    dbname="RAG",
    user="postgres",
    password="darika_2006",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Get all active chunk_ids in order
cur.execute("SELECT chunk_id FROM chunks_metadata WHERE is_active = TRUE ORDER BY chunk_id ASC")
chunk_id_mapping = [row[0] for row in cur.fetchall()]

# Save mapping
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)
MAPPING_PATH = os.path.join(INDEX_DIR, "chunk_id_mapping.pkl")

with open(MAPPING_PATH, 'wb') as f:
    pickle.dump(chunk_id_mapping, f)

print(f"✅ Created mapping with {len(chunk_id_mapping)} entries")
print(f"   Saved to: {MAPPING_PATH}")

cur.close()
conn.close()