import psycopg2

conn = psycopg2.connect( 
    dbname="RAG",
    user="postgres",
    password="darika_2006",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

print("=== DOCUMENTS TABLE ===")
cur.execute("SELECT doc_id, doc_name, allowed_roles, is_active FROM documents")
for row in cur.fetchall():
    print(f"Doc ID: {row[0]}, Name: {row[1]}, Roles: {row[2]}, Active: {row[3]}")

print("\n=== CHUNKS_METADATA TABLE ===")
cur.execute("SELECT chunk_id, doc_id, allowed_roles, is_active FROM chunks_metadata LIMIT 5")
for row in cur.fetchall():
    print(f"Chunk ID: {row[0]}, Doc ID: {row[1]}, Roles: {row[2]}, Active: {row[3]}")

print("\n=== FAISS INDEX SIZE ===")
import faiss
index = faiss.read_index("hal_index.faiss")
print(f"Total vectors in FAISS: {index.ntotal}")

cur.close()
conn.close()