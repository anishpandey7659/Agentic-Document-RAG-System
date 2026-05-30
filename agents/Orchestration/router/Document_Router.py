# services/router.py
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from core.config import GROQ_API_KEY, Tool_MODEL
from services import EmbeddingStore, PineconeEmbedder, groq_client
from Model_Memory_store.models.schemas import RouteLLM, RouteDecision

class DocumentRouter:
    """
    Two-stage router:
      1. Embedding similarity  → narrows candidates
      2. LLM reasoning         → picks the best match
    """

    SYSTEM_PROMPT = "You are a document routing agent."

    def __init__(
        self,
        groq_client: groq_client,
        embedding_store: EmbeddingStore,
        embedder: PineconeEmbedder,
        model: str = Tool_MODEL,
        top_k: int = 3
    ):
        self._client  = groq_client
        self._store   = embedding_store
        self._embedder = embedder
        self._model   = model
        self._top_k   = top_k

    # Stage 1: Embedding similarity
    def _get_candidates(self, query: str, query_embedding: List[float]) -> List[str]:
        """Returns top-k doc_ids by cosine similarity."""
        doc_embeddings = self._store.load()
        if not doc_embeddings:
            return []

        query_vec = query_embedding
        query_vec = np.array(query_vec).reshape(1, -1)

        scores = [
            {
                "doc_id": doc_id,
                "score":  float(cosine_similarity(
                    query_vec,
                    np.array(doc_emb).reshape(1, -1)
                )[0][0])
            }
            for doc_id, doc_emb in doc_embeddings.items()
        ]

        scores.sort(key=lambda x: x["score"], reverse=True)
        return [s["doc_id"] for s in scores[:self._top_k]]

    # Stage 2: LLM routing
    def _build_catalog(self, candidate_ids: List[str], memory: Dict) -> str:
        entries = []
        for doc_id in candidate_ids:
            if doc_id not in memory:
                continue
            data = memory[doc_id]
            entries.append(
                f"doc_id  : {doc_id}\n"
                f"file    : {data['metadata'].get('file_name', 'unknown')}\n"
                f"summary : {data['summary']}\n"
                f"keywords: {', '.join(data['keywords'])}"
            )
        return "\n\n".join(entries)

    def route(self, query: str, query_embedding: List[float], memory: Dict) -> List[str]:
        """Returns validated doc_ids most relevant to the query."""
        if not memory:
            raise RuntimeError("No documents in memory.")

        candidate_ids = self._get_candidates(query,query_embedding)
        print(f"[ROUTER] Embedding candidates: {candidate_ids}")

        catalog = self._build_catalog(candidate_ids, memory)

        prompt = f"""
Select the most relevant doc_ids for answering the query.
Return an empty list if none are relevant.

--- CANDIDATE DOCUMENTS ---
{catalog}
--- END DOCUMENTS ---

User Query: {query}
"""
        response = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            model=self._model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name":   "RouteDecision",
                    "schema": RouteDecision.model_json_schema()
                }
            }
        )

        content = response.choices[0].message.content
        result  = RouteDecision.model_validate_json(content)

        valid_ids = [d for d in result.doc_ids if d in memory]
        print(f"[ROUTER] Final routed docs ({len(valid_ids)}): {valid_ids}")
        return valid_ids