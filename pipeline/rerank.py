import cohere
from typing import List, Dict
from core.config import COHERE_API_KEY


class Reranker:
    """Reranks hybrid search results using Cohere."""

    def __init__(self, api_key: str = COHERE_API_KEY, model: str = "rerank-v4.0-pro"):
        self._client = cohere.ClientV2(api_key)
        self._model  = model

    def rerank(self, query: str, results: List[Dict], top_n: int = 3) -> List[Dict]:
        texts = [r["text"] for r in results]

        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=texts,
            top_n=top_n,
        )

        return [
            {
                "rank":     rank,
                "score":    round(r.relevance_score, 4),
                "text":     results[r.index]["text"],
                "doc_id":   results[r.index]["doc_id"],
                "chunk_id": results[r.index]["chunk_id"],
                "domain":   results[r.index]["domain"],
                "source":   results[r.index]["source"],
            }
            for rank, r in enumerate(response.results, start=1)
        ]