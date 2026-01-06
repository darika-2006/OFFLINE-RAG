import ollama


def generate_answer(query, docs, max_tokens=300, temp=0.3):
    """
    Generate a structured answer using Ollama Mistral 7B
    with clean bullet points, proper tables, and clear citations.
    """

    if not docs:
        return "❌ No relevant documents found."

    # ---------- Build context ----------
    context_blocks = []
    sources = []

    for i, doc in enumerate(docs[:3]):
        doc_id = f"Doc{i+1}"
        context_blocks.append(
            f"[{doc_id}] Source: {doc['source']} | Page: {doc.get('page', 'N/A')}\n{doc['text']}"
        )
        sources.append(
            f"- [{doc_id}] {doc['source']} (Page {doc.get('page', 'N/A')})"
        )

    context = "\n\n".join(context_blocks)

    # ---------- STRICT STRUCTURED PROMPT ----------
    prompt = f"""
You are an offline AI assistant.
Answer ONLY using the provided documents.
If information is missing, say so clearly.

CRITICAL FORMATTING RULES (MUST FOLLOW):
- Use concise bullet points for normal answers
- Use SHORT phrases (max 6–8 words) in table cells
- NEVER use full sentences inside tables
- ALWAYS leave ONE blank line before tables
- Use ONLY ONE table for comparison questions
- Do NOT explain inside table cells
- Do NOT include citations inside the answer body

OUTPUT FORMAT (STRICT):

### Answer
- Point 1
- Point 2
- Point 3

OR (if comparison is required)

### Comparison Table

| Aspect | Item A | Item B |
|------|--------|--------|
| Purpose | Short phrase | Short phrase |
| Audience | Short phrase | Short phrase |
| Content | Short phrase | Short phrase |
| Update | Short phrase | Short phrase |

### Sources
(List sources here)

DOCUMENTS:
{context}

QUESTION:
{query}
"""

    try:
        stream = ollama.chat(
            model="mistral:7b",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": max_tokens,
                "repeat_penalty": 1.1
            }
        )

        answer = ""
        for chunk in stream:
            answer += chunk["message"]["content"]

        # ---------- Append clean sources ----------
        answer = answer.strip()
        answer += "\n\n### Sources\n"
        answer += "\n".join(sources)

        return answer

    except Exception as e:
        return f"❌ ERROR: {str(e)}"