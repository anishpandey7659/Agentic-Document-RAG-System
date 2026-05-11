# agents/rag_agent.py

from pipeline.retriever    import Retriever
from pipeline.smart_search import SmartSearch
from services.router       import RetrievalRouter
from services.document_agent import AgentMemoryStore


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
        memory,           # your ConversationMemory
        conv_id:  str,
    ):
        self._retriever        = retriever
        self._retrieval_router = retrieval_router
        self._memory_store     = memory_store
        self._memory           = memory
        self._conv_id          = conv_id

    def run(self, query: str, stream: bool = True):
        print(f"\n[QUERY] {query}")

        # 1. Log user message
        self._memory.add_message(self._conv_id, role="user", content=query)

        # 2. Decide: retrieve or answer directly
        needs_retrieval = self._retrieval_router.should_retrieve(query)

        full_answer = ""
        sources     = None

        # 3a. Direct LLM answer
        if not needs_retrieval:
            print("\n[MODE] General answer\nAnswer:")
            for token in self._retriever.answer_normal(query, stream=True):
                print(token, end="", flush=True)
                full_answer += token

        # 3b. RAG pipeline
        else:
            print("\n[MODE] RAG retrieval\nAnswer:")
            for event in self._retriever.retrieve_and_answer(query=query, stream=True):
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
            metadata={"sources": sources} if sources else {}
        )

        return {"answer": full_answer, "sources": sources}