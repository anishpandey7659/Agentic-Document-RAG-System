import time
from typing import List, Dict
#1st ways
from agents.Orchestration.router.Document_Router import DocumentRouter
from services import PineconeEmbedder, PineconeVectorStore, AgentMemoryStore


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
    
    def search(self, query: str, query_embedding: List[float], mode: str = "dense", alpha: float = 0.7, top_k: int = 3) -> List[Dict]:

        t0 = time.perf_counter()
        t_mem = time.perf_counter()
        
        memory = self._memory_store.load_all()
        print(f"[LATENCY] memory load: {time.perf_counter() - t_mem:.4f}s")
        
        t_route = time.perf_counter()
        relevant_doc_ids = self._document_router.route(query, query_embedding, memory)
        print(f"[LATENCY] routing: {time.perf_counter() - t_route:.4f}s")
        
        if not relevant_doc_ids:
            print("[ROUTER] No relevant documents found.")
            return []
        
        t_embed = time.perf_counter()
        dense_vec = query_embedding
        sparse_vec = None

        if mode == "hybrid":
            sparse_vec = self._embedder.embed_sparse([query], input_type="query")[0]
        print(f"[LATENCY] embedding: {time.perf_counter() - t_embed:.4f}s")
        all_matches = []

        for doc_id in relevant_doc_ids:
            index_name = memory[doc_id]["vector_db"]
            t_search = time.perf_counter()
            print(f"[INFO] Searching '{index_name}' for doc '{doc_id}' ...")
            try:
                if mode == "hybrid":
                    matches = self._vector_store.search(
                        index_name=index_name,
                        dense=dense_vec,
                        sparse=sparse_vec,
                        top_k=top_k
                    )
                else:
                    matches = self._vector_store.search(
                        index_name=index_name,
                        dense=dense_vec,
                        top_k=top_k
                    )
                print(f"[LATENCY] search({doc_id}): {time.perf_counter() - t_search:.4f}s")
                all_matches.extend(matches)

            except Exception as e:
                print(f"[WARNING] Failed search '{index_name}': {e}")

        t_sort = time.perf_counter()
        all_matches.sort(key=lambda x: x["score"], reverse=True)
        print(f"[LATENCY] sort: {time.perf_counter() - t_sort:.4f}s")

        return all_matches[:top_k]

    @staticmethod
    def build_context(matches: List[Dict]) -> str:
        parts = [
            # f"[Chunk {i} | doc: {m['doc_id']} | Rank {m['rank']} | score: {m['score']}]\n{m['text']}"
            f"[Chunk {i} | doc: {m['doc_id']} |  score: {m['score']}]\n{m['text']}"
            for i, m in enumerate(matches, 1)
        ]
        return "\n\n---\n\n".join(parts)