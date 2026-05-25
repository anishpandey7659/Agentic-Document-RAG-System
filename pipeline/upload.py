# pipeline/upload.py

import os
import uuid
from typing import Optional

from core.config import INDEX_NAME

from core.dependencies import (
    Extractor,Chunker,Summarizer,
    PineconeEmbedder,EmbeddingStore,PineconeVectorStore,
    DocumentAgentFactory,AgentMemoryStore,
)
from Model_Memory_store.models import DocumentAgent
from pathlib import Path
from fastapi import  HTTPException



class UploadPipeline:
    """
    Orchestrates the full document ingestion pipeline:
    Extract → Clean → Chunk → Embed → Store → Summarize → Register
    """

    def __init__(
        self,
        extractor:       Extractor,
        chunker:         Chunker,
        summarizer:      Summarizer,
        embedder:        PineconeEmbedder,
        embedding_store: EmbeddingStore,
        vector_store:    PineconeVectorStore,
        agent_factory:   DocumentAgentFactory,
        memory_store:    AgentMemoryStore,
    ):
        self._extractor       = extractor
        self._chunker         = chunker
        self._summarizer      = summarizer
        self._embedder        = embedder
        self._embedding_store = embedding_store
        self._vector_store    = vector_store
        self._agent_factory   = agent_factory
        self._memory_store    = memory_store

    def run(
        self,
        file_path:  str,
        index_name: str,
        domain:     str,
    ) -> DocumentAgent:  

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            doc_id    = f"doc-{uuid.uuid4().hex[:8]}"
            file_name = os.path.basename(file_path)
            print(f"[INFO] Starting pipeline for '{file_name}' → doc_id: {doc_id}")

            # 1. Extract + clean
            print("[INFO] Extracting text...")
            raw_text     = self._extractor.extract(file_path)
            cleaned_text = self._extractor.clean(raw_text)
            if not cleaned_text:
                raise ValueError("Document appears to be empty after cleaning.")

            # 2. Chunk
            print("[INFO] Chunking text...")
            chunks = self._chunker.chunk(cleaned_text,file_path,{"domain": domain})
            print(f"[INFO] Created {len(chunks)} chunks.")

            # Text from document 
            texts = [doc.page_content for doc in chunks]

            # 3. Embed chunks (dense + sparse) → store in Pinecone
            print("[INFO] Generating embeddings...")
            embeddings = self._embedder.embed_both(texts, input_type="passage")

            print("[INFO] Storing in Pinecone...")
            vector_db = self._vector_store.store(
                doc_id, texts, embeddings, index_name, file_name, domain
            )

            # 4. Summarize
            print("[INFO] Summarizing document...")
            summary_data = self._summarizer.summarize(cleaned_text)

            # 5. Embed document-level representation for routing
            document_text = (
                f"Title: {file_name}\n"
                f"Domain: {domain}\n"
                f"Summary: {summary_data['summary']}\n"
                f"Keywords: {', '.join(summary_data['keywords'])}"
            )
            doc_embedding = self._embedder.embed_dense([document_text], input_type="passage")
            self._embedding_store.save(doc_id, doc_embedding)

            # 6. Register agent in memory
            agent = self._agent_factory.create(doc_id, summary_data, vector_db, file_name)
            self._memory_store.register(agent)

            print("[INFO] Pipeline complete.")
            return agent

        except Exception as e:
            print(f"[ERROR] Pipeline failed for '{file_path}': {e}")
            raise


    