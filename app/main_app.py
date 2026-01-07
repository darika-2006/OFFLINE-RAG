import streamlit as st
import requests
import psycopg2
import json

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    padding: 0.75rem;
    font-weight: 600;
}
.success-box {
    padding: 1rem;
    border-radius: 10px;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    margin: 1rem 0;
}
.error-box {
    padding: 1rem;
    border-radius: 10px;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🤖 RAG System")
st.sidebar.markdown("---")

# Backend status
col1, col2 = st.sidebar.columns(2)

with col1:
    try:
        requests.get("http://localhost:8001/", timeout=1)
        st.success("✅ Upload")
    except:
        st.error("❌ Upload")

with col2:
    try:
        requests.get("http://localhost:8000/", timeout=1)
        st.success("✅ Query")
    except:
        st.error("❌ Query")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 LLM Status")

try:
    r = requests.get("http://localhost:8000/health/llm", timeout=2)
    data = r.json()

    if data["ollama_service"] == "running":
        st.sidebar.success("✅ Ollama Running")
    else:
        st.sidebar.error("❌ Ollama Offline")
        st.sidebar.code("ollama serve")

    if data["llm_working"]:
        st.sidebar.success(f"✅ LLM: {data['llm_mode']}")
    else:
        st.sidebar.warning(f"⚠️ LLM Issue: {data['llm_mode']}")

    if data["gpu_available"]:
        gpu = data["gpu_info"]
        st.sidebar.info(f"🎮 GPU: {gpu['name']}")
        st.sidebar.metric("VRAM Free", f"{gpu['memory_free_gb']} GB")
    else:
        st.sidebar.info("💻 CPU Mode")

except:
    st.sidebar.error("❌ LLM Health Check Failed")

st.sidebar.markdown("---")

# --------------------------------------------------
# PAGE SELECT
# --------------------------------------------------
page = st.sidebar.radio("Select Action", ["📤 Upload Documents", "🔍 Query Documents"])

# ==================================================
# UPLOAD PAGE (UNCHANGED)
# ==================================================
if page == "📤 Upload Documents":
    st.title("📤 Upload Documents")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)

    with col1:
        allowed_roles = st.text_input("Allowed Roles", value="ceo")

    with col2:
        uploaded_by = st.text_input("User ID / Username", value="ad123")

    if st.button("🚀 Upload Documents"):
        if not uploaded_files:
            st.warning("⚠️ No files selected")
        else:
            try:
                files = [('files', (f.name, f.getvalue(), f.type)) for f in uploaded_files]

                r = requests.post(
                    "http://localhost:8001/upload",
                    files=files,
                    data={
                        "allowed_roles": allowed_roles.lower(),
                        "uploaded_by": uploaded_by
                    },
                    timeout=600
                )

                if r.status_code == 200:
                    data = r.json()
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Upload Successful</h3>
                        <p>Files: {data['files_uploaded']}</p>
                        <p>Chunks: {data['total_chunks']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(r.json().get("detail", "Upload failed"))

            except Exception as e:
                st.error(str(e))

# ==================================================
# QUERY PAGE (STREAMING FIXED)
# ==================================================
elif page == "🔍 Query Documents":
    st.title("🔍 Query Documents")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_area(
            "Enter your question",
            height=100,
            placeholder="What are the basic principles of flight?"
        )

    with col2:
        role = st.selectbox("Your Role", ["admin", "ceo", "manager", "user"])

    if st.button("🔍 Search"):
        if not query.strip():
            st.warning("⚠️ Enter a question")
        else:
            status_box = st.empty()
            answer_box = st.empty()
            sources_box = st.empty()

            try:
                response = requests.post(
                    "http://localhost:8000/query/stream",
                    json={"query": query, "role": role},
                    stream=True,
                    timeout=600
                )

                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")

                        if decoded.startswith("data:"):
                            payload = decoded.replace("data:", "").strip()
                            data = json.loads(payload)

                            # ---- STATUS ----
                            if data["status"] in ["searching", "found", "generating"]:
                                status_box.info(data["message"])

                            # ---- FINAL ----
                            st.write(data)

                            if data["status"] == "complete":
                                status_box.success("✅ Answer Generated")

                                answer_box.subheader("💡 Answer")
                                answer_box.info(data.get("answer", ""))

                                sources = data.get("sources", [])
                                if sources:
                                    sources_box.subheader("📚 Sources")
                                    for i, src in enumerate(sources, 1):
                                        with sources_box.expander(f"Source {i}"):
                                            st.write(src)

                                break

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out (model still generating)")
            except Exception as e:
                st.error(f"❌ Streaming error: {str(e)}")

# --------------------------------------------------
# FOOTER DB INFO
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Database Info")

try:
    conn = psycopg2.connect(
        dbname="RAG",
        user="postgres",
        password="darika_2006",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM documents WHERE is_active = TRUE")
    st.sidebar.metric("📄 Documents", cur.fetchone()[0])

    cur.execute("SELECT COUNT(*) FROM chunks_metadata WHERE is_active = TRUE")
    st.sidebar.metric("📦 Chunks", cur.fetchone()[0])

    cur.close()
    conn.close()
except:
    st.sidebar.error("❌ DB Offline")
