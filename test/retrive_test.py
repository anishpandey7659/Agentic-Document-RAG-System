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

query="what are the recent new on Agentic Ai"

# Usage
full_answer = ""
sources = None
if retrieval_router.should_retrieve(query):
        print("Answer:")
        for event in retrive.retrieve_and_answer(query=query, stream=True):
            if event["type"] == "sources":
                sources = event["sources"]

            elif event["type"] == "token":
                print(event["token"], end="", flush=True)
                full_answer += event["token"]
else:
    print("Answer:")
    for token in retrive.answer_normal(query,stream=True):
        print(token, end="", flush=True)
        full_answer += token
    print() 
    
if sources:
    print("Sources:", sources)

# python -m test.retrive_test