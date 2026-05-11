# from agents.rag_agent import RAGAgent
# from Model_Memory_store.memory.memory_manager import memory, conv_id

# # ... all your existing wiring from earlier ...

# agent = RAGAgent(
#     retriever        = retriever,
#     retrieval_router = retrieval_router,
#     memory_store     = memory_store,
#     memory           = memory,
#     conv_id          = conv_id,
# )

# if __name__ == "__main__":
#     query = "What is the recent political news in nepal"
#     agent.run(query, stream=True)