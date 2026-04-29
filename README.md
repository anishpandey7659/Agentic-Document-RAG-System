# 🤖 Agentic Document RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that lets you upload documents and ask questions against them. Unlike basic RAG systems that blindly search all documents, this system uses an **LLM-powered router** that reads document summaries and keywords to intelligently decide which documents are relevant before searching — saving cost and improving accuracy.

---

## 🏗️ Architecture

```
                        UPLOAD PIPELINE
                        ───────────────
  PDF / DOCX / TXT
        │
        ├──── Extract text
        │
        ├──── Clean + normalize
        │
        ├──── ┌─────────────────────────────────┐
        │     │  Two separate chunking paths    │
        │     │                                 │
        │     │  Vector DB chunks (400 words)   │──→ Mistral Embed ──→ Pinecone
        │     │  Summary chunks  (12,000 chars) │──→ Groq LLM      ──→ summary + keywords
        │     └─────────────────────────────────┘
        │
        └──── Register agent → system_memory.json


                        RETRIEVAL PIPELINE
                        ──────────────────
  User Query
        │
        ├──── LLM Router reads system_memory.json
        │     (summaries + keywords of all docs)
        │     → decides which doc_ids are relevant
        │
        ├──── Embed query with Mistral
        │
        ├──── Search ONLY relevant Pinecone indexes
        │
        ├──── Build context from top-K chunks
        │
        └──── Groq LLM answers from context
```

---

## ✨ Features

- **Agentic routing** — LLM reads document summaries and keywords to pick only relevant documents before searching, instead of brute-forcing all indexes
- **Mistral embeddings** — uses `mistral-embed` (1024-d vectors) for both document chunks and query embedding
- **Smart summarization** — map-reduce approach handles documents of any size; splits into chunks, summarizes each, merges results
- **Structured outputs** — Pydantic schemas enforce LLM response format, no fragile regex parsing
- **Rate limit handling** — proactive token tracking + exponential backoff retry for Groq free tier
- **Persistent memory** — `system_memory.json` stores agent metadata so documents survive process restarts
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
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

pip install -r requirements.txt
```

### requirements.txt

```
groq
mistralai
pinecone
pydantic
PyPDF2
python-docx
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
from document_upload_pipeline import upload_document_pipeline

agent = upload_document_pipeline("research_paper.pdf")

print(agent.doc_id)          # doc_a1b2c3d4
print(agent.summary)         # "This paper introduces..."
print(agent.keywords)        # ["attention", "transformer", ...]
```

### 2. Ask a question

```python
from document_retrieval_pipeline import retrieve_and_answer

answer = retrieve_and_answer(
    query="What is the attention mechanism?",
    top_k=5,
    show_chunks=True    # set False to hide raw retrieved chunks
)

print(answer)
```

### 3. See all uploaded documents

```python
from document_retrieval_pipeline import list_available_documents

list_available_documents()
```

```
=======================================================
  doc_id               keywords
=======================================================
  doc-a1b2c3d4         attention, transformer, encoder, decoder, NLP
                       Introduces Transformer architecture using self-attention...
  -------------------------------------------------------
  doc-e5f6g7h8         revenue, sales, Q3, growth, forecast
                       Q3 2024 sales report showing 23% revenue growth...
  -------------------------------------------------------
```

---

## 📁 Project Structure

```
agentic-rag/
│
├── document_upload_pipeline.py     # Upload + index documents
├── document_retrieval_pipeline.py  # Query + answer from documents
├── system_memory.json              # Auto-generated — agent registry
├── requirements.txt
└── README.md
```

---

## 🔄 How the Agent Router Works

Instead of searching every Pinecone index on every query (expensive and slow), the router asks the LLM:

```
"Here are all my documents with their summaries and keywords.
 Which ones are likely to answer this question?"
```

The LLM returns only the relevant `doc_ids`. Only those Pinecone indexes are searched.

```
10 documents uploaded
User asks: "What is multi-head attention?"

Router reads all 10 summaries + keywords
→ LLM selects: ["doc-a1b2c3d4"]   (only the transformer paper)
→ Search 1 index instead of 10    ← 90% fewer Pinecone calls
```

---

## ⚠️ Groq Rate Limits

Groq free tier has token-per-minute (TPM) limits. The pipeline handles this automatically with:

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
