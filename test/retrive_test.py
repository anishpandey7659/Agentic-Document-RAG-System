from core.dependencies import (
    retrieval_router, 
    memory_store,
    memory,embedder,retriever,memory,embedder
)
from cache.semantic_cache import SemanticCache
from cache.llmresponsecache import LLMResponseCache
from agents.rag_agent import RAGAgent
conv_id="4cfaf3fb-4a3d-4d0d-9147-70da25ed485d"
rag = RAGAgent(
        retriever,
        retrieval_router,
        memory_store,
        embedder,
        memory,
        conv_id,
        SemanticCache(),
        LLMResponseCache()
    )



import time

if __name__ == "__main__":
    start_time = time.time() 
    query = "Implement the Deep Agent in langchain and also explain it in brief?"

    conv_id="4cfaf3fb-4a3d-4d0d-9147-70da25ed485d"



    print(rag.run(query))

    end_time = time.time()
    print("Execution time:", end_time - start_time, "seconds")

    
# python -m test.retrive_test







