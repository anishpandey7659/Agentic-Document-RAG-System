import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY   = os.getenv("COHERE_API_KEY")
PINECONE_CLOUD   = "aws"
PINECONE_REGION  = "us-east-1"


EMBED_MODEL = "mistral-embed"   # 1024-dimensional vectors
Tool_MODEL   = "openai/gpt-oss-120b"  
CHAT_MODEL   = "llama-3.3-70b-versatile"
Small_MODEL  = "llama-3.1-8b-instant"   
MEMORY_FILE = "/home/anish/Documents/Agentic-RAG/memory/system_memory.json"
summarize_llm="meta-llama/llama-4-scout-17b-16e-instruct"


GROQ_TPM_LIMIT  = 131072
GROQ_TPM_BUFFER = 0.85    
TOKEN_WINDOW    = 60  