from typing import List, Dict
from config import GROQ_API_KEY, Tool_MODEL
from groq import Groq
from services import load_document_embeddings,embed_dense
from sklearn.metrics.pairwise import cosine_similarity
from Model_Memory_store.models import RouteDecision
import numpy as np


groq_client = Groq(api_key=GROQ_API_KEY)



def normalize_embedding(x):
    return np.array(x).squeeze().reshape(1, -1)

# STEP 1 → EMBEDDING ROUTER

def get_candidate_docs(query: str, top_k: int = 5):

    doc_embeddings = load_document_embeddings()

    # Query embedding
    query_embedding = embed_dense(query, input_type="query")[0]

    scores = []

    for doc_id, doc_embedding in doc_embeddings.items():

        similarity = cosine_similarity(
        normalize_embedding(query_embedding),
        normalize_embedding(doc_embedding)
        )[0][0]

        scores.append({
            "doc_id": doc_id,
            "score": float(similarity)
        })

    # Sort descending
    scores.sort(key=lambda x: x["score"], reverse=True)

    return scores[:top_k]


# STEP 2 → LLM ROUTER
def route_query_to_documents(query: str, memory: Dict) -> List[str]:

    if not memory:
        raise RuntimeError("No documents in memory.")

    # Get embedding candidates
    candidate_docs = get_candidate_docs(query, top_k=3)
    candidate_ids = [d["doc_id"] for d in candidate_docs]
    print(f"[ROUTER] Candidate docs: {candidate_ids}")

    # Build catalog ONLY for candidates
    doc_catalog = []
    for doc_id in candidate_ids:
        if doc_id not in memory:
            continue
        data = memory[doc_id]
        doc_catalog.append(
            f"doc_id: {doc_id}\n"
            f"file: {data['metadata'].get('file_name', 'unknown')}\n"
            f"summary: {data['summary']}\n"
            f"keywords: {', '.join(data['keywords'])}"
        )
    catalog_text = "\n\n".join(doc_catalog)

    # LLM Routing Prompt
    prompt = f"""
You are a document routing agent.
Select the most relevant doc_ids for answering the query.
Return empty list if none are relevant.
--- CANDIDATE DOCUMENTS ---
{catalog_text}
--- END DOCUMENTS ---

User Query:
{query}
"""

    response = groq_client.chat.completions.create(
        model=Tool_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a document routing agent."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "RouteDecision",
                "schema": RouteDecision.model_json_schema()
            }
        }
    )

    content = response.choices[0].message.content

    result = RouteDecision.model_validate_json(content)

    valid_doc_ids = [
        d for d in result.doc_ids
        if d in memory
    ]

    print(
        f"[ROUTER] Final routed docs ({len(valid_doc_ids)}): "
        f"{valid_doc_ids}"
    )

    return valid_doc_ids