import os
import re
import json
from PyPDF2 import PdfReader
import docx

# ---------- Utilities -------------

def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            pages.append((i+1, page_text))
    return pages

def load_docx_file(path):
    doc = docx.Document(path)
    full_text = "\n".join(para.text for para in doc.paragraphs)
    return [(None, full_text)]

def clean_text(text):
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(\w+)\n(\w+)", r"\1 \2", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, size=350):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])


# ---------- Main Pipeline -------------

if __name__ == "__main__":

    RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_docs")
    OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "chunks.json")

    chunks = []
    chunk_id = 0

    for fname in os.listdir(RAW_DIR):

        path = os.path.join(RAW_DIR, fname)
        print(f"Processing: {fname}")

        if fname.endswith(".pdf"):
            pages = load_pdf(path)
        elif fname.endswith(".docx"):
            pages = load_docx_file(path)
        elif fname.endswith(".txt"):
            pages = [(None, open(path).read())]
        else:
            continue  

        for page, raw_text in pages:

            cleaned = clean_text(raw_text)

            for chunk in chunk_text(cleaned, size=350):

                chunks.append({
                    "chunk_id": chunk_id,
                    "source": fname,
                    "page": page,
                    "text": chunk,
                    "access": "public" if "restricted" not in fname.lower() else "restricted"
                })

                chunk_id += 1

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"\nSaved {len(chunks)} chunks to {OUTPUT} ✅")      
