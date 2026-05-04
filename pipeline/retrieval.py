from pipeline import smart_search,build_context
from config import Tool_MODEL,GROQ_API_KEY
from groq import Groq

groq_client = Groq(api_key=GROQ_API_KEY)


def answer_query(query: str, context: str) -> str:
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
        ]
    )
    return response.choices[0].message.content.strip()

def retrieve_and_answer(
    query: str,
    top_k: int = 5,
    show_chunks: bool = False
) -> str:
    """
    Full smart retrieval pipeline:
      1. Agent reads summaries/keywords → picks relevant docs
      2. Embeds query with Mistral
      3. Searches only the relevant Pinecone indexes
      4. LLM answers from retrieved context
    """
    print(f"\n[QUERY] {query}")

    matches = smart_search(query, top_k=top_k)

    if not matches:
        return "No relevant content found for your query."

    if show_chunks:
        print("\n--- Retrieved Chunks ---")
        for m in matches:
            print(f"  [score: {m['score']} | doc: {m['doc_id']}] {m['text'][:120]}...")
        print("------------------------\n")

    context = build_context(matches)
    return answer_query(query, context)