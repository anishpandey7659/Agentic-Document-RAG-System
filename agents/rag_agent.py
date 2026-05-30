# agents/rag_agent.py
from pipeline import Retriever, SmartSearch
from agents.Orchestration.router.Retriver_Router import RetrievalRouter
from services import PineconeEmbedder, AgentMemoryStore
from cache import SemanticCache , LLMResponseCache
from Model_Memory_store.memory.memory_manager import memory
import time



class RAGAgent:
    def __init__(
        self,
        retriever:        Retriever,
        retrieval_router: RetrievalRouter,
        memory_store:     AgentMemoryStore,
        embedder:         PineconeEmbedder,
        memory,
        conv_id:          str,
        SemanticCache:    SemanticCache,
        llm_response_cache: LLMResponseCache,  # ← ADD THIS
    ):
        self._retriever           = retriever
        self._retrieval_router    = retrieval_router
        self._memory_store        = memory_store
        self._memory              = memory
        self._embedder            = embedder
        self._conv_id             = conv_id
        self._semantic_cache      = SemanticCache
        self._llm_response_cache  = llm_response_cache  # ← ADD THIS

    def run(self, query: str, mode: str = "dense", stream: bool = True):
        t0 = time.perf_counter()
        print(f"\n[QUERY] {query}")

        #  Layer 1: Semantic cache check 
        query_embedding = self._embedder.embed_dense([query], input_type="query")[0]
        semantic_cache_result = self._semantic_cache.find(query_embedding)

        if semantic_cache_result:
            print("[SEMANTIC CACHE HIT]")
            self._memory.add_message(self._conv_id, role="assistant", content=semantic_cache_result)
            return {"answer": semantic_cache_result, "sources": None}

        #  Log user message 
        self._memory.add_message(self._conv_id, role="user", content=query)

        needs_retrieval = self._retrieval_router.should_retrieve(query)
        print(f"[LATENCY] From query to query_embed to semantic_cachecheck to storemessage to retrival_route: {time.perf_counter() - t0:.4f}s")

        full_answer = ""
        sources     = None

        # Direct LLM (no retrieval — Layer 2 doesn't apply here) 
        if not needs_retrieval:
            print("\n[MODE] General answer\nAnswer:")
            for token in self._retriever.answer_normal(query, stream=stream):
                print(token, end="", flush=True)
                full_answer += token

        #  RAG pipeline 
        else:
            print("\n[MODE] RAG retrieval")

            # Step 1: retrieve chunks first (before calling LLM)
            chunks = self._retriever.retrieve_chunks(
                query=query,
                query_embedding=query_embedding,
                mode=mode
            )

            # Step 2: Layer 2 — check hash(query + chunks)
            llm_cache_result = self._llm_response_cache.find(query, chunks)

            if llm_cache_result:
                print("[LLM CACHE HIT]")
                full_answer = llm_cache_result["answer"]
                sources     = llm_cache_result["sources"]

            else:
                # Step 3: cache miss → call LLM with chunks
                print("Answer:")
                for event in self._retriever.answer_from_chunks(
                    query=query, chunks=chunks, stream=stream
                ):
                    if event["type"] == "sources":
                        sources = event["sources"]
                    elif event["type"] == "token":
                        print(event["token"], end="", flush=True)
                        full_answer += event["token"]

                # Step 4: store in Layer 2 cache
                self._llm_response_cache.store(query, chunks, full_answer, sources)

                # Step 5: store in Layer 1 cache too (for future semantic hits)
                self._semantic_cache.store(query, emb=query_embedding, answer=full_answer, chunks=chunks)

        print("\n")

        if sources:
            print("\nSources:")
            for s in sources:
                print(f"  [{s['rank']}] {s['source']} (score={s['score']})")

        self._memory.add_message(
            self._conv_id,
            role="assistant",
            content=full_answer,
            metadata={"sources": sources} if sources else None
        )

        return {"answer": full_answer, "sources": sources}