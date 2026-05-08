import cohere
from config import COHERE_API_KEY
from pipeline import smart_search

def rerank_documents(query: str, hybrid_results: list[dict],api_key=COHERE_API_KEY ,top_n: int = 3) -> list[dict]:
    co = cohere.ClientV2(api_key)

    # ✅ Extract just the text strings
    texts = [chunk['text'] for chunk in hybrid_results]

    response = co.rerank(
        model="rerank-v4.0-pro",
        query=query,
        documents=texts,
        top_n=top_n,
    )

    results = []

    for rank, r in enumerate(response.results, start=1):
        original_chunk = hybrid_results[r.index]

        results.append({
            "rank": rank,  # actual reranked position
            "score": round(r.relevance_score, 4),
            "text": original_chunk["text"],
            "doc_id": original_chunk["doc_id"],
            "chunk_id": original_chunk["chunk_id"],
        })

    return results


# --- Usage ---
# if __name__ == "__main__":
#     query="What is Agentic AI"
#     matches = smart_search(query,0.75, top_k=5)
#     results = rerank_documents(
#         query="best language for frontend?",
#         hybrid_results=matches,
#         top_n=3,
#         api_key=COHERE_API_KEY
#     )


#     for r in results:
#         print(f"Rank {r['rank']} (score={r['score']}) | {r['text']}")

# python -m pipeline.rerank