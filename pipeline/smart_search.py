from typing import List, Dict
from services.embedding.pinecone_embedder import PineconeEmbedder
from agents.Orchestration.router.Document_Router import DocumentRouter
from services.pinecone_client import PineconeVectorStore
from services.Document_agents import AgentMemoryStore


class SmartSearch:
    """
    Hybrid search across relevant documents only.
    1. Router picks relevant doc_ids from memory
    2. Embeds query once (dense + sparse)
    3. Searches only selected indexes
    4. Merges and ranks results
    """

    def __init__(
        self,
        embedder: PineconeEmbedder,
        vector_store: PineconeVectorStore,
        document_router: DocumentRouter,
        memory_store: AgentMemoryStore,
    ):
        self._embedder        = embedder
        self._vector_store    = vector_store
        self._document_router = document_router
        self._memory_store    = memory_store

    def hybrid_score_norm(self,dense: list, sparse: dict, alpha: float):
        """
        Blend dense and sparse vectors using a convex combination:
            alpha * dense + (1 - alpha) * sparse

        Args:
            dense  : list of floats — the dense embedding
            sparse : dict with 'indices' and 'values' keys
            alpha  : float in [0, 1]
                    1.0 = pure semantic (dense only)
                    0.0 = pure keyword  (sparse only)
                    0.75 = recommended starting point
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")

        scaled_dense = [v * alpha for v in dense]
        scaled_sparse = {
            "indices": sparse["indices"],
            "values":  [v * (1 - alpha) for v in sparse["values"]]
        }
        return scaled_dense, scaled_sparse


    def search(self, query: str, alpha: float = 0.7, top_k: int = 5) -> List[Dict]:
        memory = self._memory_store.load_all()

        relevant_doc_ids = self._document_router.route(query, memory)
        if not relevant_doc_ids:
            print("[ROUTER] No relevant documents found.")
            return []

        query_emb = self._embedder.embed_both([query], input_type="query")[0]
        hdense, hsparse = self.hybrid_score_norm(
            dense=query_emb["dense"],
            sparse=query_emb["sparse"],
            alpha=alpha
        )

        all_matches = []
        for doc_id in relevant_doc_ids:
            index_name = memory[doc_id]["vector_db"]
            print(f"[INFO] Searching '{index_name}' for doc '{doc_id}' ...")
            try:
                matches = self._vector_store.search(index_name, hdense, hsparse, top_k=top_k)
                all_matches.extend(matches)
            except Exception as e:
                print(f"[WARNING] Failed to search '{index_name}': {e}")

        all_matches.sort(key=lambda x: x["score"], reverse=True)
        return all_matches[:top_k]

    @staticmethod
    def build_context(matches: List[Dict]) -> str:
        parts = [
            f"[Chunk {i} | doc: {m['doc_id']} | Rank {m['rank']} | score: {m['score']}]\n{m['text']}"
            for i, m in enumerate(matches, 1)
        ]
        return "\n\n---\n\n".join(parts)