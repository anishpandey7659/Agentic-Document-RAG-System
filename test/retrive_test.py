from agents.Orchestration.router.Document_Router import  DocumentRouter
from agents.Orchestration.router.Retriver_Router import  RetrievalRouter
from services.GroqClient import groq_client 
from services.embedding.embedding_store import EmbeddingStore
from services.embedding.pinecone_embedder import PineconeEmbedder
from services.pinecone_client import PineconeClient,PineconeVectorStore
from services.Document_agents import AgentMemoryStore
from pipeline.retriever import Retriever
from pipeline.smart_search import SmartSearch
from pipeline.rerank import Reranker
from config import MEMORY_FILE
from agents.rag_agent import RAGAgent
from Model_Memory_store.memory.memory_manager import memory


store   = EmbeddingStore()
pinecone_client = PineconeClient()
embedder = PineconeEmbedder(client=pinecone_client.client)
vectorstore=PineconeVectorStore(pinecone_client)
memory_store=AgentMemoryStore()

retrieval_router = RetrievalRouter(groq_client=groq_client)
document_router  = DocumentRouter(groq_client=groq_client, embedding_store=store, embedder=embedder)

search=SmartSearch(embedder,vectorstore,document_router,memory_store)
rerank=Reranker()
retrive=Retriever(groq_client,search,rerank)

query="Hi how are you"

conv_id="4cfaf3fb-4a3d-4d0d-9147-70da25ed485d"
rag=RAGAgent(retrive,retrieval_router,memory_store,memory,conv_id)
print(rag.run(query))


# python -m test.retrive_test