import os
import ollama

def generate_answer(query, docs, max_tokens=300, temp=0.3):
    """
    Generate an answer using Ollama Mistral 7B with retrieved documents.
    """
    
    if not docs:
        return "❌ No relevant documents found."
    
    # Use top 3 documents with full context (Mistral can handle more)
    context = "\n\n".join([
        f"Document {i+1} (Source: {doc['source']}, Page: {doc.get('page', 'N/A')}):\n{doc['text']}"
        for i, doc in enumerate(docs[:3])
    ])
    
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided documents.
If the answer is not in the documents, say "I don't have enough information to answer that."

Documents:
{context}

Question: {query}

Answer: be specific and specify which document(s) you used to derive your answer. Cite sources in the format [Document X, Page Y] where applicable."""
    
    try:
        stream = ollama.chat(
            model='mistral',
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
            options={
                'temperature': 0.2,     # Even more deterministic
                'top_p': 0.9,           # Filter unlikely tokens
                'num_predict': 300,     # Max length
                'repeat_penalty': 1.1   # Avoid repetitive text
            }   
        )
        
        answer = ""
        for chunk in stream:
            answer += chunk['message']['content']
        
        return answer.strip()
    
    except Exception as e:
        return f"❌ ERROR: {str(e)}"