# 🤖 Agentic Document RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that lets you upload documents and ask questions against them. Unlike basic RAG systems that blindly search all documents, this system uses an **LLM-powered router** that reads document summaries and keywords to intelligently decide which documents are relevant before searching — saving cost and improving accuracy.

---

## 🏗️ Architecture

```
                        UPLOAD PIPELINE
                        ───────────────
  PDF / DOCX / TXT
        │
        ├──── Extract text           (pipeline/extract.py)
        │
        ├──── Clean + normalize
        │
        ├──── ┌─────────────────────────────────┐
        │     │  Two separate chunking paths    │  (pipeline/chunker.py)
        │     │                                 │
        │     │  Vector DB chunks (400 words)   │──→ Mistral Embed ──→ Pinecone
        │     │  Summary chunks  (12,000 chars) │──→ Groq LLM      ──→ summary + keywords
        │     └─────────────────────────────────┘       (pipeline/summarize_chunks.py)
        │
        └──── Register agent → memory/system_memory.json
                                    (memory/memory_manager.py)


                        RETRIEVAL PIPELINE
                        ──────────────────
  User Query
        │
        ├──── LLM Router reads system_memory.json      (services/router.py)
        │     (summaries + keywords of all docs)
        │     → decides which doc_ids are relevant
        │
        ├──── Embed query with Mistral                 (services/embedder.py)
        │
        ├──── Search ONLY relevant Pinecone indexes    (services/pinecone_client.py)
        │                                              (pipeline/docs_search.py)
        ├──── Build context from top-K chunks
        │
        └──── Groq LLM answers from context            (services/llm.py)
```

---

## ✨ Features

- **Agentic routing** — LLM reads document summaries and keywords to pick only relevant documents before searching, instead of brute-forcing all indexes
- **Mistral embeddings** — uses `mistral-embed` (1024-d vectors) for both document chunks and query embedding
- **Smart summarization** — map-reduce approach handles documents of any size; splits into chunks, summarizes each, merges results
- **Structured outputs** — Pydantic schemas enforce LLM response format, no fragile regex parsing
- **Rate limit handling** — proactive token tracking + exponential backoff retry for Groq free tier
- **Persistent memory** — `memory/system_memory.json` stores agent metadata so documents survive process restarts
- **Multi-format support** — PDF, DOCX, and TXT documents

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM (chat + routing) | [Groq](https://groq.com) — `llama-3.3-70b-versatile` |
| LLM (chunk summaries) | [Groq](https://groq.com) — `llama-3.1-8b-instant` |
| Embeddings | [Mistral AI](https://mistral.ai) — `mistral-embed` |
| Vector DB | [Pinecone](https://pinecone.io) |
| Validation | [Pydantic](https://docs.pydantic.dev) |
| PDF parsing | PyPDF2 |
| DOCX parsing | python-docx |

---

## 📦 Installation

```bash
git clone https://github.com/anishpandey7659/Agentic-Document-RAG-System.git
cd Agentic-Document-RAG-System

pip install -r requirements.txt
```

### requirements.txt

```
mistralai
PyPDF2
python-docx
langchain
langgraph
python-dotenv
pydantic
pinecone
groq
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
MISTRAL_API_KEY=your_mistral_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Or export them directly:

```bash
export GROQ_API_KEY="your_groq_api_key"
export MISTRAL_API_KEY="your_mistral_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"
```

---

## 🚀 Usage

### 1. Upload a document

```python
from pipeline.upload import upload_document_pipeline

agent = upload_document_pipeline("research_paper.pdf")

print(agent.doc_id)          # doc_a1b2c3d4
print(agent.summary)         # "This paper introduces..."
print(agent.keywords)        # ["attention", "transformer", ...]
```

### 2. Ask a question

```python
from pipeline.retrieval import retrieve_and_answer

answer = retrieve_and_answer(
    query="What is the attention mechanism?",
    top_k=5,
    show_chunks=True    # set False to hide raw retrieved chunks
)

print(answer)
```

### 3. Run via CLI

```bash
python main.py
```

---

## 📁 Project Structure

```
AGENTIC-RAG/
│
├── api/                            # API layer (if applicable)
│
├── data/                           # Sample documents for testing
│   ├── docx/
│   ├── pdf/
│   └── text_files/
│
├── memory/                         # Persistent agent registry
│   ├── memory_manager.py           # Read/write agent metadata
│   └── system_memory.json          # Auto-generated — stores all doc agents
│
├── models/                         # Pydantic schemas & data models
│   ├── __init__.py
│   └── schemas.py                  # Structured output schemas for LLM responses
│
├── pipeline/                       # Core document processing steps
│   ├── __init__.py
│   ├── chunker.py                  # Split documents into vector + summary chunks
│   ├── docs_search.py              # Search Pinecone for relevant chunks
│   ├── extract.py                  # Extract text from PDF / DOCX / TXT
│   ├── retrieval.py                # End-to-end retrieval + answer pipeline
│   ├── summarize_chunks.py         # Map-reduce summarization for large docs
│   └── upload.py                   # End-to-end upload + indexing pipeline
│
├── services/                       # External service clients
│   ├── __init__.py
│   ├── Document_agents.py          # Agent dataclass and registry logic
│   ├── embedder.py                 # Mistral embedding calls
│   ├── llm.py                      # Groq LLM calls (chat + summarization)
│   ├── pinecone_client.py          # Pinecone index management + upsert/query
│   ├── rate_limiting.py            # Token tracking + retry/backoff logic
│   └── router.py                   # LLM-powered document relevance router
│
├── .gitignore
├── config.py                       # Centralized config (model names, chunk sizes, etc.)
├── main.py                         # Entry point
├── README.md
└── requirements.txt
```

---

## 🔄 How the Agent Router Works

Instead of searching every Pinecone index on every query (expensive and slow), the router (`services/router.py`) asks the LLM:

```
"Here are all my documents with their summaries and keywords.
 Which ones are likely to answer this question?"
```

The LLM returns only the relevant `doc_ids`. Only those Pinecone indexes are searched.

```
10 documents uploaded
User asks: "What is multi-head attention?"

Router reads all 10 summaries + keywords  ← services/router.py
→ LLM selects: ["doc-a1b2c3d4"]           (only the transformer paper)
→ Search 1 index instead of 10            ← 90% fewer Pinecone calls
```

---

## ⚠️ Groq Rate Limits

Groq free tier has token-per-minute (TPM) limits. The pipeline handles this automatically in `services/rate_limiting.py` with:

- **Proactive token tracking** — monitors tokens used per 60s window
- **Auto-pause** — waits for window reset before hitting the limit
- **Retry with backoff** — catches any rate limit errors and retries

You'll see clear logs instead of silent hangs:

```
[TOKEN] Used 2,847 tokens | Total this window: 2,847/131,072
[TOKEN] Used 3,012 tokens | Total this window: 5,859/131,072
[RATE LIMIT] Token budget near limit. Waiting 34.2s for window reset...
[INFO] Resuming after reset.
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
