from typing import List, Dict
from config import EMBED_MODEL,MISTRAL_API_KEY
from mistralai.client import Mistral
from services import route_query_to_documents,search_index,load_system_memory,embed_both,hybrid_score_norm


mistral_client = Mistral(api_key=MISTRAL_API_KEY)


def embed_query(query: str) -> List[float]:
    response = mistral_client.embeddings.create(
        model=EMBED_MODEL,
        inputs=[query]
    )
    return response.data[0].embedding



def smart_search(query: str,alpha:float=0.7, top_k: int = 5) -> List[Dict]:
    """
    1. Load memory (summaries + keywords + index names)
    2. LLM router picks relevant doc_ids
    3. Embed the query once
    4. Search only the selected indexes
    5. Merge + rank results
    """
    memory = load_system_memory()

    # Step 1: Agent decides which documents to search
    relevant_doc_ids = route_query_to_documents(query, memory)

    if not relevant_doc_ids:
        print("[ROUTER] No relevant documents found for this query.")
        return []

    # Step 2: Embed query (single Mistral call)
    
    query_emb = embed_both([query], input_type="query")[0]
    hdense, hsparse = hybrid_score_norm(
            dense=query_emb["dense"],
            sparse=query_emb["sparse"],
            alpha=alpha
        )

    # Step 3: Search only the relevant indexes
    all_matches = []
    for doc_id in relevant_doc_ids:
        index_name = memory[doc_id]["vector_db"]   # ← this is how index_name is obtained
        print(f"[INFO] Searching index '{index_name}' for doc '{doc_id}' ...")

        try:
            matches = search_index(index_name, hdense, hsparse,top_k=top_k)
            all_matches.extend(matches)
        except Exception as e:
            print(f"[WARNING] Failed to search index '{index_name}': {e}")

    # Step 4: Rank all results across selected docs by score
    all_matches.sort(key=lambda x: x["score"], reverse=True)
    return all_matches[:top_k]

def build_context(matches: List[Dict]) -> str:
    parts = []
    for i, match in enumerate(matches, 1):
        parts.append(
            f"[Chunk {i} | doc: {match['doc_id']} | Rank {match['rank']} | score: {match['score']}]\n{match['text']}"
        )
    return "\n\n---\n\n".join(parts)

# print(f"Rank {m['rank']} (score={m['score']}) | {m['text']}")