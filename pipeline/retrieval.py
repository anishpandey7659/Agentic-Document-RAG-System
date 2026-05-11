from pipeline import smart_search,build_context
from .rerank import rerank_documents
from typing import List, Dict
from config import Tool_MODEL,GROQ_API_KEY,COHERE_API_KEY
from  Model_Memory_store.memory.memory_manager import memory,conv_id
from groq import Groq

groq_client = Groq(api_key=GROQ_API_KEY)


def answer_query(query: str, context: str, stream: bool = False):
    prompt = f"""
You are a helpful assistant. Answer the user's question using ONLY
the document context below. If the context does not contain enough
information, say No clearly.

--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---

User Question: {query}

Answer:
"""

    response = groq_client.chat.completions.create(
        model=Tool_MODEL,
        messages=[
            {"role": "system", "content": "You are a document Q&A assistant. Answer only from the provided context."},
            {"role": "user", "content": prompt}
        ],
        stream=stream,
    )

    if stream:
        return response  
    return response.choices[0].message.content.strip()


def retrieve_and_answer(
    query: str,
    alpha: float = 0.7,
    top_k: int = 6,
    show_chunks: bool = False,
    stream: bool = False,
):
    print(f"\n[QUERY] {query}")

    matches = smart_search(query, alpha, top_k=top_k)

    if not matches:
        if stream:
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "token": "No relevant content found for your query."}
            return
        return {"answer": "No relevant content found for your query.", "sources": []}

    reranked_matches = rerank_documents(query, hybrid_results=matches, api_key=COHERE_API_KEY, top_n=3)

    if not reranked_matches:
        if stream:
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "token": "No relevant content found for your query."}
            return
        return {"answer": "No relevant content found for your query.", "sources": []}

    if show_chunks:
        print("\n--- Retrieved Chunks ---")
        for m in reranked_matches:
            print(f"Rank {m['rank']} (score={m['score']}) | {m['text']}")
        print("------------------------\n")

    context = build_context(reranked_matches)

    if stream:
        # yield sources first so UI can render them immediately
        yield {"type": "sources", "sources": reranked_matches}
        for chunk in answer_query(query, context, stream=True):
            token = chunk.choices[0].delta.content or ""
            if token:
                yield {"type": "token", "token": token}
    else:
        answer = answer_query(query, context)
        return {"answer": answer, "sources": reranked_matches}
    

def answer_normal(query: str, stream: bool = False):
    response = groq_client.chat.completions.create(
        model=Tool_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Give accurate answers."},
            {"role": "user", "content": query}
        ],
        stream=True,
    )
    full_response = ""
    if stream:
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                yield delta
    else:
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta

        return full_response.strip()