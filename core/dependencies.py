# core/dependencies.py

#  Clients 
from services import PineconeClient, groq_client
from Model_Memory_store.memory.memory_manager import memory

#  Services 
from services import (
    PineconeEmbedder, PineconeVectorStore,
    EmbeddingStore, AgentMemoryStore,
    DocumentAgentFactory,
)

#  Routers 
from agents.Orchestration.router.Document_Router import DocumentRouter
from agents.Orchestration.router.Retriver_Router import RetrievalRouter

#  Pipeline 
from pipeline import (
    Extractor, Chunker, Summarizer, UploadPipeline,
    SmartSearch, Reranker, Retriever
)

pinecone_client = PineconeClient()

# Services (shared across both pipelines)
embedder        = PineconeEmbedder(client=pinecone_client.client)
embedding_store = EmbeddingStore()
vector_store    = PineconeVectorStore(pinecone_client)
agent_factory   = DocumentAgentFactory()
memory_store    = AgentMemoryStore()

# Upload Pipeline 
extractor  = Extractor()
chunker    = Chunker()
summarizer = Summarizer(groq_client=groq_client)

upload_pipeline = UploadPipeline(
    extractor       = extractor,
    chunker         = chunker,
    summarizer      = summarizer,
    embedder        = embedder,
    embedding_store = embedding_store,
    vector_store    = vector_store,
    agent_factory   = agent_factory,
    memory_store    = memory_store,
)

#  RAG Pipeline
retrieval_router = RetrievalRouter(groq_client=groq_client)
document_router  = DocumentRouter(
                       groq_client=groq_client,
                       embedding_store=embedding_store,
                       embedder=embedder
                   )
smart_search     = SmartSearch(embedder, vector_store, document_router, memory_store)
reranker         = Reranker()
retriever        = Retriever(groq_client=groq_client, smart_search=smart_search, reranker=reranker)