from services.GroqClient import groq_client
from services.pinecone_client import PineconeClient, PineconeVectorStore
from services.embedding.pinecone_embedder  import PineconeEmbedder
from services.embedding.embedding_store    import EmbeddingStore
from services.Document_agents import DocumentAgentFactory, AgentMemoryStore
from agents.Orchestration.router.Document_Router import  DocumentRouter
from agents.Orchestration.router.Retriver_Router import RetrievalRouter 
from pipeline.extractor import Extractor
from pipeline.chunker import Chunker
from pipeline.summarizer import Summarizer
from pipeline.upload import UploadPipeline
from pipeline.smart_search import SmartSearch
from pipeline.rerank import Reranker
from pipeline.retriever import Retriever
from config import INDEX_NAME


# Clients 

pinecone_client = PineconeClient()

# Services 
embedder = PineconeEmbedder(client=pinecone_client.client) 
embedding_store = EmbeddingStore()
vector_store    = PineconeVectorStore(pinecone_client)
agent_factory   = DocumentAgentFactory()
memory_store    = AgentMemoryStore()

# Pipeline 
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


retrieval_router = RetrievalRouter(groq_client=groq_client)
document_router  = DocumentRouter(groq_client=groq_client, embedding_store=embedding_store, embedder=embedder)
smart_search     = SmartSearch(embedder, vector_store, document_router, memory_store)
reranker         = Reranker()
retriever        = Retriever(groq_client=groq_client, smart_search=smart_search, reranker=reranker)


if __name__ == "__main__":
    agent = upload_pipeline.run(
        file_path  = "/home/anish/Downloads/paper.pdf",
        index_name = INDEX_NAME,
        domain     = "Python introduction"
    )
    print(f"\n  doc_id  : {agent.doc_id}")
    print(f"  file    : {agent.metadata['file_name']}")
    print(f"  summary : {agent.summary}")
    print(f"  keywords: {agent.keywords}")
    print(f"  vector  : {agent.vector_db}")



# python -m test.upload_test