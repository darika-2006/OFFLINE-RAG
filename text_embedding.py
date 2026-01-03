from sentence_transformers import  SentenceTransformer #sentence transformers to used for embedding
import pypdfium2 as pdfium #fastest way to extract text from a PDF
import faiss  #a similarity search mechanism (not a traditional DB)
import numpy as np  #python list can't be used as a input for faiss , so converted into numpy array
import json  #For metadata(chunk) storage , used for citation
import psycopg2  #for DB connection
import os  #for OS
    
#Dimension : (28, 384) Mini - but works for large data
# model=SentenceTransformer("all-MiniLM-L6-v2")


# new code with DB connected :
conn = psycopg2.connect( 
    dbname="RAG",#have a doubt here , what is dbname ?
    user="postgres", #what is user ? i don't know what to fill , so i just gave my name
    password="Bala@2007",
    host="localhost",
    port="5432"
)
cur=conn.cursor()
cur.execute("SELECT current_database(), current_schema()")
print(cur.fetchone())


def chunk_list(no_of_files):
    chunks=[]
    for i in range(1,no_of_files+1):
        pdfName=input(f"Enter the document name {i} : ")
        # pdfId=input("Enter the document ID :")
        pdfPath=input("Enter the Path of the Document :")
        allowed_roles=input("Enter the roles separated by spaces :").lower()
        uploaded_by="admin"
        sql_query1 = """
        INSERT INTO documents(doc_name, doc_path, allowed_roles, uploaded_by) VALUES (%s, %s, %s, %s)RETURNING doc_id"""
        cur.execute(sql_query1, (pdfName, pdfPath, allowed_roles, uploaded_by))
        conn.commit()
        doc_id=cur.fetchone()[0]
        temp=chunk_pdf_with_page(pdfPath)
        for chunk in temp:
                chunk["doc_id"] = doc_id
                chunk["allowed_roles"] = allowed_roles
                chunks.append(chunk)
        # chunks.extend(temp)
    return chunks

def chunk_pdf_with_page(pdf_path, max_words=250, overlap=50):
    pdf = pdfium.PdfDocument(pdf_path)
    chunks = []

    for page_num in range(len(pdf)):
        page = pdf.get_page(page_num)
        text_page = page.get_textpage()
        page_text = text_page.get_text_range()
        text_page.close()
        page.close()

        words = page_text.split()
        start = 0

        while start < len(words):
            end = start + max_words
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "page_number": page_num + 1,  # human-readable
                "text": chunk_text
            })

            start = end - overlap
            if overlap >= max_words:
                break

    pdf.close()
    return chunks


def build_metadata(chunks,base_chunk_id):
        for i,chunk in enumerate(chunks):
            chunk_id=base_chunk_id+i
            sql_query2="INSERT INTO chunks_metadata(chunk_id,doc_id,page_number,chunk_text,allowed_roles) VALUES(%s,%s,%s,%s,%s)"
            # chunk_id = base_chunk_id + i
            cur.execute(sql_query2,(chunk_id,chunk["doc_id"],chunk["page_number"],chunk["text"],chunk["allowed_roles"]))
            conn.commit()

def delete_metadata():
    doc_name=input("Enter the document name you want to delete :")
    doc_id="SELECT doc_id FROM documents WHERE doc_name=%s" 
    cur.execute(doc_id,(doc_name,))
    result=cur.fetchone()
    if  result:
        sql_query4="UPDATE documents SET is_active=FALSE WHERE doc_id=%s"
        sql_query5="UPDATE chunks_metadata SET is_active=FALSE WHERE doc_id=%s"
        cur.execute(sql_query4,(result[0],))
        cur.execute(sql_query5,(result[0],))
        conn.commit()
    else:
        print("Result is not propelry fetched")

no_of_files=int(input("Enter the number of files you're going to upload :"))
chunks=chunk_list(no_of_files)

if not chunks:
    raise ValueError("No text extracted from PDF")
model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts, normalize_embeddings=True)

# print(embeddings.shape)
dimension = embeddings.shape[1]

if os.path.exists("hal_index.faiss"):
    index = faiss.read_index("hal_index.faiss")
else:
    index = faiss.IndexFlatL2(dimension)

base_chunk_id = index.ntotal
index.add(embeddings)

build_metadata(chunks,base_chunk_id)
faiss.write_index(index, "hal_index.faiss")

