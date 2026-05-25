# agents/rag_agent.py
from pipeline import Retriever, SmartSearch
from agents.Orchestration.router.Retriver_Router import RetrievalRouter
from services import PineconeEmbedder, AgentMemoryStore
from cache import SemanticCache , LLMResponseCache, EmbeddingCache



class RAGAgent:
    """
    Top-level agent that:
    1. Decides whether retrieval is needed
    2. Routes to RAG pipeline or direct LLM answer
    3. Tracks conversation memory
    """

    def __init__(
        self,
        retriever:        Retriever,
        retrieval_router: RetrievalRouter,
        memory_store:     AgentMemoryStore,
        embedder:         PineconeEmbedder,
        memory,          # your ConversationMemory
        conv_id:  str,
        SemanticCache: SemanticCache,
    ):
        self._retriever        = retriever
        self._retrieval_router = retrieval_router
        self._memory_store     = memory_store
        self._memory           = memory
        self._embedder        = embedder
        self._conv_id          = conv_id
        self._semantic_cache   = SemanticCache

    def run(self, query: str,mode: str = "dense",stream: bool = True):
        print(f"\n[QUERY] {query}")

        query_embedding = self._embedder.embed_dense([query], input_type="query")[0]
        semantic_cache_result = self._semantic_cache.find(query_embedding)

        if semantic_cache_result:
            print("[SEMANTIC CACHE HIT] response")
            self._memory.add_message(self._conv_id, role="assistant", content=semantic_cache_result)
            return {"answer": semantic_cache_result, "sources": None}

        # 1. Log user message
        self._memory.add_message(self._conv_id, role="user", content=query)

        # 2. Decide: retrieve or answer directly
        needs_retrieval = self._retrieval_router.should_retrieve(query)

        full_answer = ""
        sources     = None

        # 3a. Direct LLM answer
        if not needs_retrieval:
            print("\n[MODE] General answer\nAnswer:")
            for token in self._retriever.answer_normal(query, stream=stream):
                print(token, end="", flush=True)
                full_answer += token

        # 3b. RAG pipeline
        else:
            print("\n[MODE] RAG retrieval\nAnswer:")
            for event in self._retriever.retrieve_and_answer(query=query, query_embedding=query_embedding, mode=mode, stream=stream):
                if event["type"] == "sources":
                    sources = event["sources"]
                elif event["type"] == "token":
                    print(event["token"], end="", flush=True)
                    full_answer += event["token"]

        print("\n")

        # 4. Print sources if any
        if sources:
            print("\nSources:")
            for s in sources:
                print(f"  [{s['rank']}] {s['source']} (score={s['score']})")
        
        # 5. Log assistant response
        self._memory.add_message(
            self._conv_id,
            role="assistant",
            content=full_answer,
            metadata={"sources": sources} if sources else None
        )

        return {"answer": full_answer, "sources": sources}