# 🤖 Agentic-RAG

A production-grade **Agentic Retrieval-Augmented Generation (RAG)** system that intelligently routes queries, retrieves relevant documents, reranks results, and generates accurate answers — built with a clean, fully class-based architecture.

---

## ✨ Features

- 🔍 **Hybrid Search** — combines dense + sparse embeddings for superior retrieval
- 🧠 **Agentic Routing** — LLM decides whether to retrieve or answer directly
- 📄 **Document Routing** — embedding similarity + LLM picks the most relevant documents
- 🏆 **Reranking** — Cohere reranker refines results before generation
- 🗂️ **Conversation Memory** — Supabase-backed conversation + document agent memory
- 📚 **Multi-format Ingestion** — supports PDF, DOCX, and TXT
- ⚡ **Streaming Support** — token-by-token streaming for real-time responses
- 🛡️ **Rate Limiting** — proactive TPM tracking with exponential backoff retry

---

## 🏗️ Architecture

```
User Query
    │
    ▼
agents/Orchestration/router/Retriver_Router.py   ← Should we retrieve at all?
    │
    ├── NO  → api/answer_llm.py                  (direct LLM answer)
    │
    └── YES → agents/Orchestration/router/Document_Router.py
                  │
                  ├── Stage 1: Embedding similarity  (narrow candidates)
                  └── Stage 2: LLM routing           (pick best doc_ids)
                              │
                              ▼
                       pipeline/smart_search.py
                              │
                              ├── embed_both()  [dense + sparse]
                              └── PineconeVectorStore.search()
                                          │
                                          ▼
                                 pipeline/rerank.py  (Cohere)
                                          │
                                          ▼
                                pipeline/retriever.py → Final Answer
```

---

## 📁 Project Structure

```
AGENTIC-RAG/
│
├── agents/
│   ├── __init__.py
│   ├── rag_agent.py                          # Top-level orchestrator
│   ├── Orchestration/
│   │   ├── router/
│   │   │   ├── Document_Router.py            # Two-stage doc routing (embedding + LLM)
│   │   │   └── Retriver_Router.py            # Decides retrieve vs direct answer
│   └── State/                                # Agent state management
│
├── api/
│   ├── __init__.py
│   ├── answer_llm.py                         # Direct LLM answer endpoint
│   └── upload_file.py                        # Document upload endpoint
│
├── data/                                     # Raw + processed documents
│
├── Model_Memory_store/
│   ├── memory/
│   │   ├── doc_embeddings.pkl                # Cached document-level embeddings
│   │   ├── memory_manager.py                 # Supabase memory layer (MemoryLayer class)
│   │   └── system_memory.json                # Registered document agents
│   └── models/
│       └── __init__.py                       # Pydantic schemas (DocumentAgent, DocumentSummary, RouteDecision ...)
│
├── pipeline/
│   ├── __init__.py
│   ├── chunker.py                            # Word-based overlapping chunker
│   ├── extractor.py                          # Extract + clean text (PDF/DOCX/TXT)
│   ├── rerank.py                             # Cohere reranker
│   ├── retriever.py                          # Orchestrates search → rerank → answer
│   ├── smart_search.py                       # Hybrid search across selected indexes
│   ├── summarizer.py                         # Chunk summarization + keyword extraction
│   └── upload.py                             # Full document ingestion pipeline
│
├── services/
│   ├── __init__.py
│   ├── embedding/
│   │   ├── embedding_service.py              # Unified embedding facade
│   │   ├── embedding_store.py                # Pickle-based embedding persistence
│   │   ├── mistral_embedder.py               # Mistral text embeddings
│   │   └── pinecone_embedder.py              # Pinecone dense + sparse embeddings
│   ├── Document_agents.py                    # DocumentAgentFactory + AgentMemoryStore
│   ├── GroqClient.py                         # GroqClient with rate limiting + retry
│   ├── llm.py                                # LLM call helpers
│   ├── mistralai_client.py                   # Mistral client singleton
│   ├── pinecone_client.py                    # PineconeClient + PineconeVectorStore
│   └── rate_limiting.py                      # TokenBudget
│
├── test/
│   ├── __init__.py
│   ├── retrive_test.py
│   └── upload_test.py
│
├── .env                                      # API keys (never commit)
├── .gitignore
├── config.py                                 # Loads .env, exports all constants
├── main.py                                   # Wires all dependencies + entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | [Groq](https://groq.com) (LLaMA 3) |
| Embeddings | [Mistral](https://mistral.ai) + [Pinecone Inference](https://docs.pinecone.io) |
| Vector DB | [Pinecone](https://pinecone.io) (hybrid dense + sparse) |
| Reranking | [Cohere](https://cohere.com) rerank-v4.0-pro |
| Memory | [Supabase](https://supabase.com) (PostgreSQL) |
| Orchestration | Custom agentic routing (class-based) |
| File Parsing | PyPDF2, python-docx |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
# Groq
GROQ_API_KEY=your_groq_api_key
GROQ_TPM_LIMIT=6000
GROQ_TPM_BUFFER=0.85
TOKEN_WINDOW=60

# Mistral
MISTRAL_API_KEY=your_mistral_api_key
EMBED_MODEL=mistral-embed

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
INDEX_NAME=your_index_name

# Cohere
COHERE_API_KEY=your_cohere_api_key

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# App
MEMORY_FILE=Model_Memory_store/memory/system_memory.json
EMBEDDING_FILE=Model_Memory_store/memory/doc_embeddings.pkl
Tool_MODEL=llama3-8b-8192
summarize_llm=llama3-8b-8192
```

### 5. Set up Supabase tables

Run this SQL in your Supabase dashboard:

```sql
CREATE TABLE USERS (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE CONVERSATIONS (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES USERS(id),
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE MESSAGES (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES CONVERSATIONS(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);
```

---

## 📖 Usage

### Upload a document

```python
from main import upload_pipeline

agent = upload_pipeline.run(
    file_path  = "data/my_document.pdf",
    index_name = "my-index",
    domain     = "Finance"
)

print(f"doc_id  : {agent.doc_id}")
print(f"summary : {agent.summary}")
print(f"keywords: {agent.keywords}")
```

### Query the system

```python
from main import rag_agent

# Streaming
result = rag_agent.run("What does the document say about risk?", stream=True)

# Non-streaming
result = rag_agent.run("Summarize the key findings.", stream=False)
print(result["answer"])
print(result["sources"])
```

---

## 🔄 Full Pipeline Flow

### Ingestion
```
File (PDF / DOCX / TXT)
    → Extractor.extract() + clean()
    → Chunker.chunk()
    → PineconeEmbedder.embed_both()          ← dense + sparse per chunk
    → PineconeVectorStore.store()             ← upsert to Pinecone index
    → Summarizer.summarize()                  ← chunk → merge → keywords
    → PineconeEmbedder.embed_dense()          ← document-level embedding
    → EmbeddingStore.save()                   ← cached to doc_embeddings.pkl
    → DocumentAgentFactory.create()
    → AgentMemoryStore.register()             ← saved to system_memory.json
```

### Retrieval
```
Query
    → Retriver_Router.should_retrieve()       ← retrieve or answer directly?
    → Document_Router.route()
         ├── _get_candidates()                ← cosine similarity on doc embeddings
         └── LLM routing prompt               ← picks best doc_ids
    → SmartSearch.search()
         ├── embed_both(query)                ← hybrid query embedding
         └── PineconeVectorStore.search()     ← per selected Pinecone index
    → Reranker.rerank()                       ← Cohere rerank-v4.0-pro
    → SmartSearch.build_context()
    → GroqClient.complete_text()              ← final answer generation
```

---

## 🧩 Key Design Principles

### 1. Single Responsibility
Every class has one job. `Reranker` only reranks. `Chunker` only chunks. `MemoryLayer` only talks to Supabase.

### 2. Dependency Injection
No hidden globals. Every class receives its dependencies through the constructor:

```python
retriever = Retriever(
    groq_client  = groq,
    smart_search = smart_search,
    reranker     = reranker,
)
```

### 3. One Wiring Point
`main.py` is the only place that knows how everything connects. Swap Cohere for another reranker? Change one line in `main.py`.

### 4. Layered Architecture

| Layer | Folder | Responsibility |
|---|---|---|
| Agent | `agents/` | Decisions + orchestration |
| Pipeline | `pipeline/` | Data flow (ingest, search, rank, answer) |
| Services | `services/` | External API clients (Groq, Pinecone, Mistral, Cohere) |
| API | `api/` | HTTP request handling |
| Memory | `Model_Memory_store/` | Conversation + document persistence |

### 5. No yield + return Mix
Streaming and non-streaming are always separate code paths — delegated down to `GroqClient.stream_tokens()` and `GroqClient.complete_text()` so the outer methods stay clean.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Follow the class-based architecture — no loose functions in `pipeline/` or `services/`
4. Open a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
